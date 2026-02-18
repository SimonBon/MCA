from copy import deepcopy

_base_ = [
    '../_base_/default.py',  
    '../_base_/augmentations_virtualpatients.py',
    '../_base_/train_cfg.py',
    '../_base_/val_cfg.py',
    '../_datasets_/CODEX_cHL.py', 
    '../_models_/ATT.py',
]

batch_size = 256
num_workers = 16
mask_patch = True

_base_.val_augmentation[0].size = _base_.cutter_size
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]

_base_.train_augmentation_vp1[-2].size = _base_.cutter_size
_base_.train_augmentation_vp2[-2].size = _base_.cutter_size
_base_.train_pipeline[0].transforms = [_base_.train_augmentation_vp1, _base_.train_augmentation_vp2]

train_dataset = deepcopy(_base_.dataset)
train_dataset.update(_base_.dataset_kwargs)

train_dataset['used_indicies'] = _base_.train_indicies
train_dataset['pipeline'] = _base_.train_pipeline
train_dataset['mask_patch'] = mask_patch

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers, 
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=train_dataset,
)

_base_.custom_hooks[0].train_indicies = _base_.train_indicies
_base_.custom_hooks[0].val_indicies = _base_.val_indicies
_base_.custom_hooks[0].pipeline = _base_.val_pipeline
_base_.custom_hooks[0].dataset_kwargs = _base_.dataset_kwargs

_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.backbone.input_size = _base_.cutter_size
_base_.model.neck.in_channels = _base_.n_markers * _base_.features_per_marker

work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS/CODEX_cHL_ATT_MASK_VP'