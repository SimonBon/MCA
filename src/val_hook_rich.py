from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from copy import deepcopy
from .utils import cast_data
from tqdm import tqdm
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, precision_score,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed — skipping UMAP. Install with: pip install umap-learn")


@HOOKS.register_module()
class EvaluateModelRich(Hook):
    """
    Drop-in replacement for EvaluateModel that runs five complementary
    evaluations on the frozen representation:

        1. Linear probe  — standard SSL benchmark (logistic regression)
        2. k-NN          — parameter-free, tests feature similarity structure
        3. k-means       — unsupervised clustering quality (NMI / ARI)
        4. Neighbourhood purity — fraction of k nearest neighbours sharing
                                  the same cell-type label
        5. Silhouette score — cluster compactness in cosine space
        6. UMAP plot     — 2-D visualisation coloured by cell type (requires
                           umap-learn; skipped gracefully if missing)

    metrics.json keeps the same top-level 'train'/'val' keys as EvaluateModel
    (from the linear probe) so existing downstream code still works.
    """

    def __init__(
            self,
            dataset_kwargs: dict,
            train_indicies,
            val_indicies,
            pipeline,
            priority='VERY_LOW',
            epochs=1000,
            annotation_map=None,
            max_samples=None,
            knn_k=15,
            silhouette_max_samples=10_000):

        super().__init__()

        self.priority = priority
        self.train_indicies = train_indicies
        self.val_indicies = val_indicies
        self.epochs = epochs
        self._max_samples = max_samples if max_samples is not None else float('inf')
        self.knn_k = knn_k
        self.silhouette_max_samples = silhouette_max_samples

        base_dataset = dict(type='MCIDataset', pipeline=pipeline)
        base_dataset.update(dataset_kwargs)
        if annotation_map:
            base_dataset['annotation_map'] = annotation_map

        print(base_dataset)

        base_dataloader = dict(
            batch_size=32,
            num_workers=16,
            sampler=dict(type='DefaultSampler', shuffle=True),
            collate_fn=dict(type='default_collate'),
            drop_last=False,
            dataset=None,
        )

        self.train_dataloader_cfg = deepcopy(base_dataloader)
        self.train_dataloader_cfg['dataset'] = deepcopy(base_dataset)
        self.train_dataloader_cfg['dataset']['used_indicies'] = self.train_indicies

        self.val_dataloader_cfg = deepcopy(base_dataloader)
        self.val_dataloader_cfg['dataset'] = deepcopy(base_dataset)
        self.val_dataloader_cfg['dataset']['used_indicies'] = self.val_indicies

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _extract_features(self, model, dataloader, desc):
        features, labels_str, sample_ids = [], [], []
        for batch in tqdm(dataloader, desc=desc):
            feats = model(cast_data(batch['inputs'], model.device), mode='tensor')
            features.extend(feats[0].detach().cpu().numpy())
            labels_str.extend(list(batch['data_samples']['annotation'][0]))
            sample_ids.extend(list(batch['data_samples']['sample_id'][0]))
            if len(features) >= self._max_samples:
                break
        return (
            np.array(features).squeeze(),
            np.array(labels_str),
            np.array(sample_ids),
        )

    @staticmethod
    def _top2_balanced_accuracy(y_true, top2_preds, n_classes):
        recalls = []
        for c in range(n_classes):
            mask = y_true == c
            if mask.sum() == 0:
                continue
            correct = np.sum([y_true[i] in top2_preds[i] for i in np.where(mask)[0]])
            recalls.append(correct / mask.sum())
        return float(np.mean(recalls))

    # ──────────────────────────────────────────────────────────────────────
    # Main evaluation
    # ──────────────────────────────────────────────────────────────────────

    def after_train(self, runner):
        model = runner.model
        model.eval()
        work_dir = runner.work_dir

        train_dl = runner.build_dataloader(self.train_dataloader_cfg)
        val_dl   = runner.build_dataloader(self.val_dataloader_cfg)
        print(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

        # ── Feature extraction ─────────────────────────────────────────────
        train_feats, train_labels_str, train_ids = self._extract_features(
            model, train_dl, "Extracting train features")
        val_feats, val_labels_str, val_ids = self._extract_features(
            model, val_dl, "Extracting val features")

        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels_str)
        val_labels   = le.transform(val_labels_str)
        classes      = list(le.classes_)
        n_classes    = len(classes)
        print(f"\nClasses ({n_classes}): {classes}")

        metrics = {
            'classes': classes,
            'n_classes': n_classes,
            'feature_dim': int(train_feats.shape[1]),
        }

        # ── 1. Linear Probe ────────────────────────────────────────────────
        print("\n=== 1. Linear Probe ===")
        clf = LogisticRegression(
            solver='lbfgs', penalty='l2', max_iter=self.epochs,
            class_weight='balanced', C=10, n_jobs=2, verbose=1,
        )
        clf.fit(train_feats, train_labels)

        train_proba_lr = clf.predict_proba(train_feats)
        val_proba_lr   = clf.predict_proba(val_feats)
        train_pred_lr  = train_proba_lr.argmax(axis=1)
        val_pred_lr    = val_proba_lr.argmax(axis=1)
        train_top2_lr  = np.argsort(train_proba_lr, axis=1)[:, -2:]
        val_top2_lr    = np.argsort(val_proba_lr,   axis=1)[:, -2:]

        metrics['linear_probe'] = {
            'train': {
                'top1_accuracy':          float(accuracy_score(train_labels, train_pred_lr)),
                'top2_accuracy':          float(np.mean([train_labels[i] in train_top2_lr[i] for i in range(len(train_labels))])),
                'top1_balanced_accuracy': float(balanced_accuracy_score(train_labels, train_pred_lr)),
                'top2_balanced_accuracy': self._top2_balanced_accuracy(train_labels, train_top2_lr, n_classes),
                'precision':              float(precision_score(train_labels, train_pred_lr, average='weighted')),
                'f1':                     float(f1_score(train_labels, train_pred_lr, average='weighted')),
                'n_samples':              len(train_feats),
            },
            'val': {
                'top1_accuracy':          float(accuracy_score(val_labels, val_pred_lr)),
                'top2_accuracy':          float(np.mean([val_labels[i] in val_top2_lr[i] for i in range(len(val_labels))])),
                'top1_balanced_accuracy': float(balanced_accuracy_score(val_labels, val_pred_lr)),
                'top2_balanced_accuracy': self._top2_balanced_accuracy(val_labels, val_top2_lr, n_classes),
                'precision':              float(precision_score(val_labels, val_pred_lr, average='weighted')),
                'f1':                     float(f1_score(val_labels, val_pred_lr, average='weighted')),
                'n_samples':              len(val_feats),
            },
        }
        print(f"  Val balanced acc: {metrics['linear_probe']['val']['top1_balanced_accuracy']:.4f}")

        # ── 2. k-NN ────────────────────────────────────────────────────────
        # weights='distance' softens majority-class dominance compared to
        # uniform voting, though k-NN has no direct class_weight equivalent.
        print(f"\n=== 2. k-NN (k={self.knn_k}, cosine, distance-weighted) ===")
        knn = KNeighborsClassifier(n_neighbors=self.knn_k, metric='cosine',
                                   weights='distance', n_jobs=-1)
        knn.fit(train_feats, train_labels)
        train_pred_knn = knn.predict(train_feats)
        val_pred_knn   = knn.predict(val_feats)

        metrics['knn'] = {
            'k': self.knn_k,
            'train': {
                'top1_accuracy':          float(accuracy_score(train_labels, train_pred_knn)),
                'top1_balanced_accuracy': float(balanced_accuracy_score(train_labels, train_pred_knn)),
                'f1':                     float(f1_score(train_labels, train_pred_knn, average='weighted')),
            },
            'val': {
                'top1_accuracy':          float(accuracy_score(val_labels, val_pred_knn)),
                'top1_balanced_accuracy': float(balanced_accuracy_score(val_labels, val_pred_knn)),
                'f1':                     float(f1_score(val_labels, val_pred_knn, average='weighted')),
            },
        }
        print(f"  Val balanced acc: {metrics['knn']['val']['top1_balanced_accuracy']:.4f}")

        # ── 3. MLP Probe ───────────────────────────────────────────────────
        # MLPClassifier has no class_weight param — use sample_weight instead,
        # computed the same way sklearn's 'balanced' mode works internally.
        print("\n=== 3. MLP Probe (256-hidden, relu, balanced sample weights) ===")
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            max_iter=self.epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        train_sample_weights = compute_sample_weight('balanced', train_labels)
        mlp.fit(train_feats, train_labels, sample_weight=train_sample_weights)
        train_pred_mlp = mlp.predict(train_feats)
        val_pred_mlp   = mlp.predict(val_feats)

        metrics['mlp_probe'] = {
            'hidden_layer_sizes': (256,),
            'train': {
                'top1_accuracy':          float(accuracy_score(train_labels, train_pred_mlp)),
                'top1_balanced_accuracy': float(balanced_accuracy_score(train_labels, train_pred_mlp)),
                'f1':                     float(f1_score(train_labels, train_pred_mlp, average='weighted')),
                'n_samples':              len(train_feats),
            },
            'val': {
                'top1_accuracy':          float(accuracy_score(val_labels, val_pred_mlp)),
                'top1_balanced_accuracy': float(balanced_accuracy_score(val_labels, val_pred_mlp)),
                'f1':                     float(f1_score(val_labels, val_pred_mlp, average='weighted')),
                'n_samples':              len(val_feats),
            },
        }
        mlp_gap = (metrics['mlp_probe']['val']['top1_balanced_accuracy']
                   - metrics['linear_probe']['val']['top1_balanced_accuracy'])
        print(f"  Val balanced acc: {metrics['mlp_probe']['val']['top1_balanced_accuracy']:.4f}  "
              f"(gap vs linear: {mlp_gap:+.4f})")

        # ── 5. k-means → NMI / ARI ────────────────────────────────────────
        print(f"\n=== 5. k-means Clustering (k={n_classes}) ===")
        all_feats  = np.concatenate([train_feats, val_feats])
        all_labels = np.concatenate([train_labels, val_labels])
        km = MiniBatchKMeans(n_clusters=n_classes, n_init=10, random_state=42, max_iter=300)
        km.fit(all_feats)
        val_clusters = km.predict(val_feats)
        all_clusters = km.predict(all_feats)

        metrics['clustering'] = {
            'method': f'MiniBatchKMeans(k={n_classes}, n_init=10)',
            'val': {
                'nmi': float(normalized_mutual_info_score(val_labels,  val_clusters)),
                'ari': float(adjusted_rand_score(val_labels, val_clusters)),
            },
            'all': {
                'nmi': float(normalized_mutual_info_score(all_labels,  all_clusters)),
                'ari': float(adjusted_rand_score(all_labels, all_clusters)),
            },
        }
        print(f"  Val NMI: {metrics['clustering']['val']['nmi']:.4f}  |  "
              f"Val ARI: {metrics['clustering']['val']['ari']:.4f}")

        # ── 6. Neighbourhood Purity ────────────────────────────────────────
        print(f"\n=== 6. Neighbourhood Purity (k={self.knn_k}) ===")
        nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(val_feats)
        _, nn_idx = nbrs.kneighbors(val_feats)
        # nn_idx[:, 0] is the point itself — skip it
        neighbor_labels = val_labels[nn_idx[:, 1:]]            # (N, k)
        mean_purity = float(np.mean(neighbor_labels == val_labels[:, None]))

        per_class_purity = {}
        for c, cls in enumerate(classes):
            mask = val_labels == c
            if mask.sum() == 0:
                continue
            per_class_purity[cls] = float(np.mean(neighbor_labels[mask] == c))

        metrics['neighbourhood_purity'] = {
            'k': self.knn_k,
            'mean': mean_purity,
            'per_class': per_class_purity,
        }
        print(f"  Mean neighbourhood purity: {mean_purity:.4f}")

        # ── 7. Silhouette Score ────────────────────────────────────────────
        print(f"\n=== 7. Silhouette Score (cosine, "
              f"max {self.silhouette_max_samples} samples) ===")
        n_sil   = min(self.silhouette_max_samples, len(val_feats))
        sil_idx = np.random.default_rng(42).choice(len(val_feats), n_sil, replace=False)
        sil     = float(silhouette_score(val_feats[sil_idx], val_labels[sil_idx],
                                         metric='cosine'))
        metrics['silhouette'] = {'score': sil, 'n_samples': n_sil, 'metric': 'cosine'}
        print(f"  Silhouette: {sil:.4f}")

        # ── 8. Confusion Matrix (linear probe) ────────────────────────────
        val_pred_str_lr = le.inverse_transform(val_pred_lr)
        val_cm = confusion_matrix(val_labels_str, val_pred_str_lr,
                                  labels=classes, normalize='true')
        fig, ax = plt.subplots(figsize=(max(8, n_classes), max(7, n_classes - 1)))
        ConfusionMatrixDisplay(val_cm, display_labels=classes).plot(
            ax=ax, cmap='Blues', values_format='.2f', xticks_rotation=45)
        ax.set_title('Confusion Matrix — Linear Probe (Val)')
        plt.tight_layout()
        plt.savefig(f'{work_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        # also save as JSON
        def _cm_to_dict(cm, cls_list):
            return {
                'classes': cls_list,
                'matrix': {
                    true_cls: {pred_cls: round(float(cm[i, j]), 4)
                               for j, pred_cls in enumerate(cls_list)}
                    for i, true_cls in enumerate(cls_list)
                },
                'per_class_recall': {cls: round(float(cm[i, i]), 4)
                                     for i, cls in enumerate(cls_list)},
            }

        with open(f'{work_dir}/confusion_matrix_val.json', 'w') as f:
            json.dump(_cm_to_dict(val_cm, classes), f, indent=2)

        # ── 9. UMAP ───────────────────────────────────────────────────────
        if HAS_UMAP:
            print("\n=== 7. UMAP ===")
            reducer = umap_lib.UMAP(
                n_components=2, metric='cosine',
                n_neighbors=15, min_dist=0.1,
                random_state=42, verbose=True,
            )
            # Fit on train to keep val structure honest
            reducer.fit(train_feats)
            val_emb = reducer.transform(val_feats)

            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.get_cmap('tab20', n_classes)
            for i, cls in enumerate(classes):
                mask = val_labels == i
                ax.scatter(val_emb[mask, 0], val_emb[mask, 1],
                           s=1, alpha=0.4, color=cmap(i), label=cls, rasterized=True)
            ax.legend(markerscale=6, bbox_to_anchor=(1.02, 1),
                      loc='upper left', fontsize=8, frameon=False)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Val features — UMAP (coloured by cell type)')
            plt.tight_layout()
            plt.savefig(f'{work_dir}/umap.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved UMAP to {work_dir}/umap.png")

            np.savez_compressed(f'{work_dir}/umap_embeddings.npz',
                                embedding=val_emb,
                                labels_num=val_labels,
                                labels_str=val_labels_str)
            metrics['umap'] = {'saved': True, 'n_neighbors': 15,
                               'min_dist': 0.1, 'metric': 'cosine'}
        else:
            print("\nSkipping UMAP — install with: pip install umap-learn")
            metrics['umap'] = {'saved': False, 'reason': 'umap-learn not installed'}

        # ── Save feature arrays ────────────────────────────────────────────
        np.savez_compressed(
            f'{work_dir}/val_results.npz',
            features=val_feats, labels_str=val_labels_str,
            labels_num=val_labels, sample_ids=val_ids,
            top1_pred_lr=val_pred_lr, top1_pred_knn=val_pred_knn,
            classes=le.classes_,
        )
        np.savez_compressed(
            f'{work_dir}/train_results.npz',
            features=train_feats, labels_str=train_labels_str,
            labels_num=train_labels, sample_ids=train_ids,
            top1_pred_lr=train_pred_lr, top1_pred_knn=train_pred_knn,
            classes=le.classes_,
        )

        # ── Save metrics.json ──────────────────────────────────────────────
        # Mirror top-level 'train'/'val' keys from linear probe for backwards compat
        metrics['train'] = metrics['linear_probe']['train']
        metrics['val']   = metrics['linear_probe']['val']

        with open(f'{work_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # ── Summary print ──────────────────────────────────────────────────
        lp  = metrics['linear_probe']['val']
        mlp = metrics['mlp_probe']['val']
        knn = metrics['knn']['val']
        cl  = metrics['clustering']['val']
        print(f"""
╔══════════════════════════════════════════════════════╗
║              Evaluation Summary (Val)                ║
╠══════════════════════════════════════════════════════╣
║  Linear Probe  bal-acc  {lp['top1_balanced_accuracy']:.4f}   F1 {lp['f1']:.4f}      ║
║  MLP Probe     bal-acc  {mlp['top1_balanced_accuracy']:.4f}   F1 {mlp['f1']:.4f}  gap {mlp_gap:+.4f} ║
║  k-NN (k={self.knn_k:2d})    bal-acc  {knn['top1_balanced_accuracy']:.4f}   F1 {knn['f1']:.4f}      ║
║  Clustering    NMI      {cl['nmi']:.4f}   ARI {cl['ari']:.4f}         ║
║  Nbhd purity           {mean_purity:.4f}                        ║
║  Silhouette            {sil:.4f}                        ║
╚══════════════════════════════════════════════════════╝
""")
        print(f"Saved metrics to {work_dir}/metrics.json")
