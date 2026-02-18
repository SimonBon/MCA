from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from copy import deepcopy
from mmengine.registry import DATASETS
from .utils import cast_data
from tqdm import tqdm
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# import torch
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from torch import nn
# from collections import Counter
# from MIDL26.MIDL26.evaluate import base_dataset, train_pipeline, val_pipeline
# from MIDL26.MIDL26.src.utils import train, evaluate, base_dataloader
# from pathlib import Path

@HOOKS.register_module()
class EvaluateModel(Hook):
    
    def __init__(
            self, 
            dataset_kwargs: dict, 
            train_indicies, 
            val_indicies, 
            pipeline, 
            short=False, 
            priority='VERY_LOW', 
            epochs=1000): 
        
        super().__init__()
        
        self.dataset_kwargs = dataset_kwargs

        self.priority=priority
        self.train_indicies=train_indicies
        self.val_indicies=val_indicies
        self.short=short
        self.epochs=epochs
        
        base_dataset = dict(
            type='MCIDataset',
            pipeline=pipeline
        )
        base_dataset.update(self.dataset_kwargs)
        
        print(base_dataset)
        
        base_dataloader = dict(
            batch_size=32,
            num_workers=16, 
            sampler=dict(type='DefaultSampler', shuffle=True),
            collate_fn=dict(type='default_collate'),
            drop_last=False,
            dataset=None,
        )

        self.train_dataset = deepcopy(base_dataset)
        self.train_dataset['used_indicies'] = self.train_indicies
        self.train_dataloader = deepcopy(base_dataloader)
        self.train_dataloader['dataset'] = self.train_dataset
    
        self.val_dataset = deepcopy(base_dataset)
        self.val_dataset['used_indicies'] = self.val_indicies
        self.val_dataloader = deepcopy(base_dataloader)
        self.val_dataloader['dataset'] = self.val_dataset


    def after_train(self, runner):
        
        model = runner.model
        model.eval()
        
        train_dataloader = runner.build_dataloader(self.train_dataloader)
        val_dataloader = runner.build_dataloader(self.val_dataloader)
        
        print(len(train_dataloader), len(val_dataloader))
        
        # Collect all data
        train_features, train_labels_str, train_sample_ids = [], [], []
        for batch in tqdm(train_dataloader, desc="Extracting train features"):
            feats = model(cast_data(batch['inputs'], model.device), mode='tensor')
            train_features.extend(feats[0].detach().cpu().numpy())
            train_labels_str.extend(list(batch['data_samples']['annotation'][0]))
            train_sample_ids.extend(list(batch['data_samples']['sample_id'][0]))
            if len(train_features) > 25_000: 
                if self.short:
                    break
        
        train_features = np.array(train_features).squeeze()
        train_labels_str = np.array(train_labels_str)
        train_sample_ids = np.array(train_sample_ids)

        val_features, val_labels_str, val_sample_ids = [], [], []
        for batch in tqdm(val_dataloader, desc="Extracting val features"):
            feats = model(cast_data(batch['inputs'], model.device), mode='tensor')
            val_features.extend(feats[0].detach().cpu().numpy())
            val_labels_str.extend(list(batch['data_samples']['annotation'][0]))
            val_sample_ids.extend(list(batch['data_samples']['sample_id'][0]))
            if len(val_features) > 25_000: 
                if self.short:
                    break
            
        val_features = np.array(val_features).squeeze()
        val_labels_str = np.array(val_labels_str)
        val_sample_ids = np.array(val_sample_ids)
        
        # Encode labels
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels_str)
        val_labels = label_encoder.transform(val_labels_str)
        
        print(f"\nLabel classes: {label_encoder.classes_}")
        print(f"Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

        # Train classifier
        clf = LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            max_iter=self.epochs,
            class_weight='balanced',
            verbose=1,
            C=10,
            n_jobs=2
        )
        
        clf.fit(train_features, train_labels)
        
        # Get probabilities and predictions
        train_proba = clf.predict_proba(train_features)
        val_proba = clf.predict_proba(val_features)
        
        # Top-1 predictions
        train_pred_num = train_proba.argmax(axis=1)
        val_pred_num = val_proba.argmax(axis=1)
        
        # Top-2 predictions
        train_top2 = np.argsort(train_proba, axis=1)[:, -2:]
        val_top2 = np.argsort(val_proba, axis=1)[:, -2:]
        
        # Calculate metrics...
        train_top2_acc = np.mean([train_labels[i] in train_top2[i] for i in range(len(train_labels))])
        val_top2_acc = np.mean([val_labels[i] in val_top2[i] for i in range(len(val_labels))])
        
        def top2_balanced_accuracy(y_true, top2_preds, n_classes):
            recalls = []
            for c in range(n_classes):
                mask = (y_true == c)
                if mask.sum() == 0:
                    continue
                correct = np.sum([y_true[i] in top2_preds[i] for i in np.where(mask)[0]])
                recalls.append(correct / mask.sum())
            return np.mean(recalls)
        
        train_top2_bal_acc = top2_balanced_accuracy(train_labels, train_top2, len(label_encoder.classes_))
        val_top2_bal_acc = top2_balanced_accuracy(val_labels, val_top2, len(label_encoder.classes_))
        
        train_pred_str = label_encoder.inverse_transform(train_pred_num)
        val_pred_str = label_encoder.inverse_transform(val_pred_num)
        
        # Standard metrics
        train_acc = accuracy_score(train_labels, train_pred_num)
        train_bal_acc = balanced_accuracy_score(train_labels, train_pred_num)
        train_precision = precision_score(train_labels, train_pred_num, average='weighted')
        train_f1 = f1_score(train_labels, train_pred_num, average='weighted')
        
        val_acc = accuracy_score(val_labels, val_pred_num)
        val_bal_acc = balanced_accuracy_score(val_labels, val_pred_num)
        val_precision = precision_score(val_labels, val_pred_num, average='weighted')
        val_f1 = f1_score(val_labels, val_pred_num, average='weighted')
        
        # Print metrics...
        print(f"\n=== Train Set Performance ===")
        print(f"Top-1 Accuracy: {train_acc:.4f}")
        print(f"Top-2 Accuracy: {train_top2_acc:.4f}")
        print(f"Top-1 Balanced Accuracy: {train_bal_acc:.4f}")
        print(f"Top-2 Balanced Accuracy: {train_top2_bal_acc:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"F1 Score: {train_f1:.4f}")
        
        print(f"\n=== Validation Set Performance ===")
        print(f"Top-1 Accuracy: {val_acc:.4f}")
        print(f"Top-2 Accuracy: {val_top2_acc:.4f}")
        print(f"Top-1 Balanced Accuracy: {val_bal_acc:.4f}")
        print(f"Top-2 Balanced Accuracy: {val_top2_bal_acc:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"F1 Score: {val_f1:.4f}")

        # Confusion matrices
        train_cm = confusion_matrix(train_labels_str, train_pred_str, labels=label_encoder.classes_, normalize='true')
        val_cm = confusion_matrix(val_labels_str, val_pred_str, labels=label_encoder.classes_, normalize='true')

        # Plot...
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        disp_train = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=label_encoder.classes_)
        disp_train.plot(ax=axes[0], cmap='Blues', values_format='.2f', xticks_rotation=45)
        axes[0].set_title('Train Confusion Matrix')
        
        disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=label_encoder.classes_)
        disp_val.plot(ax=axes[1], cmap='Blues', values_format='.2f', xticks_rotation=45)
        axes[1].set_title('Validation Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{runner.work_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved confusion matrix to {runner.work_dir}/confusion_matrix.png")

        # ============================================
        # SAVE CONFUSION MATRICES AS JSON
        # ============================================
        classes = list(label_encoder.classes_)

        def cm_to_dict(cm, classes):
            return {
                'classes': classes,
                'matrix': {
                    true_cls: {
                        pred_cls: round(float(cm[i, j]), 4)
                        for j, pred_cls in enumerate(classes)
                    }
                    for i, true_cls in enumerate(classes)
                },
                'per_class_recall': {
                    cls: round(float(cm[i, i]), 4)
                    for i, cls in enumerate(classes)
                }
            }

        with open(f'{runner.work_dir}/confusion_matrix_train.json', 'w') as f:
            json.dump(cm_to_dict(train_cm, classes), f, indent=2)

        with open(f'{runner.work_dir}/confusion_matrix_val.json', 'w') as f:
            json.dump(cm_to_dict(val_cm, classes), f, indent=2)

        print(f"Saved confusion matrix JSON to {runner.work_dir}/confusion_matrix_{{train,val}}.json")
        
        # ============================================
        # SAVE TRAIN SPLIT (NUMPY ONLY)
        # ============================================
        
        np.savez_compressed(
            f'{runner.work_dir}/train_results.npz',
            features=train_features,
            labels_str=train_labels_str,
            labels_num=train_labels,
            sample_ids=train_sample_ids,
            top1_pred_num=train_pred_num,
            top1_pred_str=train_pred_str,
            top2_pred_num=train_top2,
            logits=train_proba,  # All class probabilities (n_samples x n_classes)
            classes=label_encoder.classes_
        )
        print(f"Saved train results to {runner.work_dir}/train_results.npz")
        
        # ============================================
        # SAVE VAL SPLIT (NUMPY ONLY)
        # ============================================
        
        np.savez_compressed(
            f'{runner.work_dir}/val_results.npz',
            features=val_features,
            labels_str=val_labels_str,
            labels_num=val_labels,
            sample_ids=val_sample_ids,
            top1_pred_num=val_pred_num,
            top1_pred_str=val_pred_str,
            top2_pred_num=val_top2,
            logits=val_proba,  # All class probabilities (n_samples x n_classes)
            classes=label_encoder.classes_
        )
        print(f"Saved val results to {runner.work_dir}/val_results.npz")
        
        # ============================================
        # SAVE METRICS JSON
        # ============================================
        
        metrics = {
            'train': {
                'top1_accuracy': float(train_acc),
                'top2_accuracy': float(train_top2_acc),
                'top1_balanced_accuracy': float(train_bal_acc),
                'top2_balanced_accuracy': float(train_top2_bal_acc),
                'precision': float(train_precision),
                'f1': float(train_f1),
                'n_samples': len(train_features)
            },
            'val': {
                'top1_accuracy': float(val_acc),
                'top2_accuracy': float(val_top2_acc),
                'top1_balanced_accuracy': float(val_bal_acc),
                'top2_balanced_accuracy': float(val_top2_bal_acc),
                'precision': float(val_precision),
                'f1': float(val_f1),
                'n_samples': len(val_features)
            },
            'classes': list(label_encoder.classes_),
            'n_classes': len(label_encoder.classes_),
            'feature_dim': int(train_features.shape[1])
        }
        
        with open(f'{runner.work_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics to {runner.work_dir}/metrics.json")