from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.colors import to_rgb
from pathlib import Path
import torch
import pandas as pd

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler, default_collate

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

base_dataset = dict(
    type='MultiChannelDataset',
    h5_file_path="/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/dataset/cHL_CODEX.h5",
    patch_size=24,
    shuffle=False,
    in_memory=None,
    markers_to_use=None,
    split=[0.7, 0.1, 0.2],
    image_key='IMAGES',
    pipeline=None,
    used_split=None,
    mask_image=None,
    classes_to_ignore=None
)

base_dataloader = dict(
    batch_size=256,
    num_workers=16, 
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=None
)

view_pipelines = [
    [
        dict(type='C_RandomFlip', prob=0.5, horizontal=True, vertical=True),
        dict(type='C_RandomAffine', angle=(0, 360), scale=(9/10, 10/9), shift=(0, 0), order=1),
        dict(type='RandomIntensity', low=9/10, high=10/9, clip=True),
        dict(type='RandomNoise', mean=(0, 0), std=(0, 0.01), clip=True),
        dict(type='CentralCutter', size=18),
    ]
]

train_pipeline = [
    dict(type='MultiView', num_views=[1], transforms=view_pipelines),
    dict(type='PackSelfSupInputs', meta_keys=['annotation', 'dataset_idx']),
]
    
val_pipeline = [
    dict(type='MultiView', num_views=[1], transforms=[dict(type='CentralCutter', size=18)]),
    dict(type='PackSelfSupInputs', meta_keys=['annotation', 'dataset_idx']),
]

def create_colormap(segmentation_value, color_value):
    
    color_value = np.array(color_value)
    if color_value.ndim == 1:
        color_value = color_value[:, None]
        
    max_id = segmentation_value.max()
    if color_value.shape[1] == 3:
        colors = np.zeros((max_id + 1, 3), dtype=float)   # index = segmentation ID, 0 stays black
    else:
        colors = np.zeros((max_id + 1, 1), dtype=float)   # index = segmentation ID, 0 stays black

    for seg_id, color in zip(segmentation_value, color_value):
        
        colors[seg_id] = color
        
    return np.array(colors)

def plot_segmentation_colored(segmentation, colors, color_mapping=None):

    # seg_rgb is your colored mask
    seg_rgb = colors[segmentation]

    plt.figure(figsize=(11, 10))
    plt.imshow(seg_rgb)
    plt.axis("off")

    # Build legend entries
    if color_mapping is not None:
        legend_handles = []
        for celltype, color in celltype_colors.items():
            legend_handles.append(
                Patch(facecolor=to_rgb(color), edgecolor='none', label=celltype)
            )

        plt.legend(
            handles=legend_handles,
            title="Cell types",
            bbox_to_anchor=(0.5, 1.05),
            loc="lower center",
            ncol=5,
        )

    plt.tight_layout()
    plt.show()
    
def find_latest_timestamp_folder(work_dir):
    work_dir = Path(work_dir)
    timestamp_folders = [p for p in work_dir.iterdir() if p.is_dir()]
    latest = max(timestamp_folders, key=lambda p: p.stat().st_mtime)
    return latest

def load_checkpoint(
    work_dir, 
    device=None, 
    get_dataset=False, 
    dataset_pipeline=None, 
    additional_keys=[],
    in_memory=False,
    used_split=None,
    dataloader_kwargs=dict(),
    h5_file_path=None,
    image_key=None,
    mask_image=None):
    
    work_dir = Path(work_dir)

    with open(work_dir / 'last_checkpoint', 'r') as f:
        latest_model = f.readlines()[0]

    print(f'Loading: {latest_model}')
    checkpoint = torch.load(latest_model, map_location='cpu')
    model_weights = checkpoint['state_dict']
    latest_config = find_latest_timestamp_folder(work_dir) / 'vis_data/config.py'
    
    print(f'Loading Model from {latest_config}')

    cfg = Config.fromfile(latest_config)

    model = MODELS.build(cfg.model)
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    print("Model loaded successfully.")
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Device for computation: {device}')
        
    model.to(device)
    
    if not get_dataset:
        return dict(model=model, dataset=None, dataloader=None, config=cfg)

    dataset = copy(base_dataset)
    dataset['markers_to_use'] = cfg['train_dataloader']['dataset']['markers_to_use'] if 'markers_to_use' in cfg['train_dataloader']['dataset'] else None
    if h5_file_path is not None:
        dataset['h5_file_path'] = h5_file_path
    if dataset_pipeline is not None:
        dataset['pipeline'] = dataset_pipeline
    else:
        dataset['pipeline'] = val_pipeline
        
    if mask_image is not None:
        dataset['mask_image'] = mask_image
    if image_key is not None:
        dataset['image_key'] = image_key

    if used_split is not None:
        dataset['used_split'] = used_split
    dataset['additional_keys'] = additional_keys
    dataset['in_memory'] = in_memory
    
    dataset = DATASETS.build(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=DefaultSampler(dataset, shuffle=False),
        collate_fn=default_collate,
        **dataloader_kwargs
    )

    return dict(model=model, dataset=dataset, dataloader=dataloader, config=cfg) 


def train(model, classifier, train_dl, label_encoder, device, criterion, epochs=10, logger=None):

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_losses = []
    for epoch in range(epochs):
        
        model.eval()
        classifier.train()
        
        model.to(device)
        classifier.to(device)
        
        running_train_loss = 0.0
        running_correct = 0
        running_total = 0

        train_iter = tqdm(train_dl, desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False)
        
        if logger is not None:
            logger.info(f"Train Epoch {epoch + 1}/{epochs}")
            
        for j, batch in enumerate(train_iter):
            imgs = [batch['inputs'][0].float().to(device)]
            labels_str = [s.annotation for s in batch['data_samples']]
            labels = torch.tensor(label_encoder.transform(labels_str)).to(device)

            feats = model(imgs)[0].squeeze()
            logits = classifier(feats)
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            running_correct += correct
            running_total += labels.size(0)
            running_acc = running_correct / running_total

            current_lr = optimizer.param_groups[0]["lr"]
            train_iter.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{running_acc:.4f}",
                lr=f"{current_lr:.6f}"
            )
            
            if (logger is not None) and (j%10==0):
                logger.info(f"Train Epoch {epoch + 1}/{epochs}: [{j}/{len(train_dl)}] loss={loss.item():.4f} acc={running_acc:.4f}")
            # Step scheduler ONCE per epoch

            train_loss = loss.item()
            train_losses.append(train_loss)
            
        scheduler.step()

    return classifier, train_losses

def evaluate(model, classifier, val_dl, label_encoder, device, criterion):
    
    model.eval()
    classifier.eval()
    
    preds, gts = [], []
    val_iter = tqdm(val_dl, leave=False)
    with torch.no_grad():
        for batch in val_iter:
            imgs = [batch['inputs'][0].float().to(device)]
            labels_str = [s.annotation for s in batch['data_samples']]
            labels = torch.tensor(label_encoder.transform(labels_str)).to(device)
            
            feats = model(imgs)[0].squeeze()
            logits = classifier(feats)
            
            loss = criterion(logits, labels)

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())

            val_iter.set_postfix(loss=loss.item())

    cm = confusion_matrix(gts, preds, normalize="true")
    bal_acc = balanced_accuracy_score(gts, preds)
    acc = accuracy_score(gts, preds)
    
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    
    labels = label_encoder.classes_
    cm = confusion_matrix(label_encoder.inverse_transform(gts), label_encoder.inverse_transform(preds), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f'Bal Acc: {bal_acc:.4f} Acc: {acc:.4f}')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    return classifier, bal_acc, acc, cm, cm_df, fig
    
from typing import Mapping, Optional, Sequence, Union
from mmengine.structures import BaseDataElement
  
def cast_data(data, device, non_blocking=True):
    """Copying data to the target device.

    Args:
        data (dict): Data returned by ``DataLoader``.

    Returns:
        CollatedResult: Inputs and data sample at target device.
    """
    if isinstance(data, Mapping):
        return {key: cast_data(data[key], device) for key in data}
    elif isinstance(data, (str, bytes)) or data is None:
        return data
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        # namedtuple
        return type(data)(*(cast_data(sample, device) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, Sequence):
        return type(data)(cast_data(sample, device) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, (torch.Tensor, BaseDataElement)):
        return data.to(device, non_blocking=non_blocking)
    else:
        return data

import os
import inspect 

def get_current_file():
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        filename = caller_frame.f_globals.get('__file__')
        
        if filename:
            return os.path.basename(filename)
        
        # Fallback: try to get from stack
        for frame_info in inspect.stack():
            if frame_info.filename != '<stdin>' and frame_info.filename != '<ipython-input-...>':
                return os.path.basename(frame_info.filename)
    finally:
        del frame
    
    return None