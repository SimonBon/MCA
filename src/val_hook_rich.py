from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from copy import deepcopy
from .utils import cast_data
from tqdm import tqdm
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, precision_score,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
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
    Drop-in replacement for EvaluateModel with five complementary evaluations:

        1. Linear probe     — standard SSL benchmark (logistic regression)
        2. k-NN             — parameter-free, tests feature similarity structure
        3. k-means          — unsupervised clustering quality (NMI / ARI)
        4. Neighbourhood purity — fraction of k nearest neighbours sharing
                                  the same cell-type label
        5. Sample integration — how well patient samples are mixed in embedding space
        6. UMAP plot        — 2-D visualisation coloured by cell type (requires
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
            epochs=2000,
            annotation_map=None,
            max_samples=None,
            knn_k=15,
            lisi_k=90,
            silhouette_max_samples=10_000,
            n_jobs=4):

        super().__init__()

        self.priority = priority
        self.train_indicies = train_indicies
        self.val_indicies = val_indicies
        self.epochs = epochs
        self._max_samples = max_samples if max_samples is not None else float('inf')
        self.knn_k = knn_k
        self.lisi_k = lisi_k
        self.silhouette_max_samples = silhouette_max_samples
        self.n_jobs = n_jobs

        base_dataset = dict(type='MCIDataset', pipeline=pipeline)
        base_dataset.update(dataset_kwargs)
        if annotation_map:
            base_dataset['annotation_map'] = annotation_map

        print(base_dataset)

        base_dataloader = dict(
            batch_size=32,
            num_workers=16,
            collate_fn=dict(type='default_collate'),
            drop_last=False,
            dataset=None,
        )

        self.train_dataloader_cfg = deepcopy(base_dataloader)
        self.train_dataloader_cfg['sampler'] = dict(type='DefaultSampler', shuffle=True)
        self.train_dataloader_cfg['dataset'] = deepcopy(base_dataset)
        self.train_dataloader_cfg['dataset']['used_indicies'] = self.train_indicies

        self.val_dataloader_cfg = deepcopy(base_dataloader)
        self.val_dataloader_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)
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
    # Loss curve plotting
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _plot_loss_curve(work_dir, smooth_window=50, zoom_frac=0.5):
        """Read all scalars.json under work_dir and save loss_curve.png.

        Layout (3 rows):
          Top    — full training loss (log y) + LR on twin axis
          Middle — last zoom_frac of training (linear y) for fine detail
          Bottom — VICReg component losses (inv / var / cov), last zoom_frac
        """
        import glob, os

        # collect all scalars.json files, merge and sort by step
        records = []
        for path in glob.glob(os.path.join(work_dir, '**/scalars.json'), recursive=True):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        if not records:
            print(f"  No scalars.json found under {work_dir} — skipping loss curve")
            return

        records.sort(key=lambda r: r.get('step', 0))
        steps    = np.array([r['step']     for r in records], dtype=float)
        loss     = np.array([r.get('loss',     np.nan) for r in records])
        loss_inv = np.array([r.get('loss_inv', np.nan) for r in records])
        loss_var = np.array([r.get('loss_var', np.nan) for r in records])
        loss_cov   = np.array([r.get('loss_cov',   np.nan) for r in records])
        loss_align = np.array([r.get('loss_align', np.nan) for r in records])
        lr         = np.array([r.get('lr',         np.nan) for r in records])

        def smooth(arr, w):
            kernel = np.ones(w) / w
            padded = np.pad(arr, (w // 2, w - w // 2 - 1), mode='edge')
            return np.convolve(padded, kernel, mode='valid')

        loss_s     = smooth(loss,     smooth_window)
        loss_inv_s = smooth(loss_inv, smooth_window)
        loss_var_s = smooth(loss_var, smooth_window)
        loss_cov_s   = smooth(loss_cov,   smooth_window)
        loss_align_s = smooth(loss_align, smooth_window)

        n       = len(steps)
        zoom_i  = int(n * (1 - zoom_frac))
        alpha_r = 0.15
        zs      = steps[zoom_i:]

        has_align = not np.all(np.isnan(loss_align)) and np.nanmax(loss_align) > 0
        n_comp = 4 if has_align else 3

        fig = plt.figure(figsize=(14, 13))
        gs  = fig.add_gridspec(3, n_comp, hspace=0.50, wspace=0.35,
                               height_ratios=[1, 1, 1])
        ax_full = fig.add_subplot(gs[0, :])
        ax_zoom = fig.add_subplot(gs[1, :])
        ax_inv  = fig.add_subplot(gs[2, 0])
        ax_var  = fig.add_subplot(gs[2, 1])
        ax_cov  = fig.add_subplot(gs[2, 2])
        ax_aln  = fig.add_subplot(gs[2, 3]) if has_align else None

        # ── Row 0: full curve, log y ───────────────────────────────────────
        ax = ax_full
        ax.plot(steps, loss,   color='steelblue', alpha=alpha_r, linewidth=0.5)
        ax.plot(steps, loss_s, color='steelblue', linewidth=1.8, label='total loss (smoothed)')
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss (log scale)', color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.set_title('Full training — loss (log) + LR')
        ax2 = ax.twinx()
        ax2.plot(steps, lr, color='tomato', linewidth=1.2, linestyle='--', alpha=0.8, label='LR')
        ax2.set_ylabel('learning rate', color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        warmup_end = steps[np.argmax(lr)]
        ax.axvspan(steps[0], warmup_end, alpha=0.08, color='orange')

        # ── Row 1: zoomed total loss, linear y ────────────────────────────
        ax = ax_zoom
        zls = loss_s[zoom_i:]
        ax.plot(zs, loss[zoom_i:], color='steelblue', alpha=alpha_r, linewidth=0.5)
        ax.plot(zs, zls,           color='steelblue', linewidth=1.8)
        lo, hi = np.nanpercentile(zls, 1), np.nanpercentile(zls, 99)
        pad = (hi - lo) * 0.15
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('iteration')
        ax.set_ylabel('total loss')
        ax.set_title(f'Last {int(zoom_frac*100)}% of training — total loss (linear, clipped)')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        # ── Row 2: each component on its own subplot / y-axis ─────────────
        def _comp_subplot(ax, raw, smoothed, label, color):
            ax.plot(zs, raw[zoom_i:],      color=color, alpha=alpha_r, linewidth=0.5)
            ax.plot(zs, smoothed[zoom_i:], color=color, linewidth=1.8)
            lo, hi = np.nanpercentile(smoothed[zoom_i:], 1), np.nanpercentile(smoothed[zoom_i:], 99)
            pad = max((hi - lo) * 0.20, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('iteration')
            ax.grid(axis='y', linestyle=':', alpha=0.5)

        _comp_subplot(ax_inv, loss_inv, loss_inv_s, f'invariance  (last {int(zoom_frac*100)}%)', 'C0')
        _comp_subplot(ax_var, loss_var, loss_var_s, f'variance    (last {int(zoom_frac*100)}%)', 'C1')
        _comp_subplot(ax_cov, loss_cov, loss_cov_s, f'covariance  (last {int(zoom_frac*100)}%)', 'C2')
        if has_align:
            _comp_subplot(ax_aln, loss_align, loss_align_s, f'alignment   (last {int(zoom_frac*100)}%)', 'C3')
        ax_inv.set_ylabel('loss')

        out = os.path.join(work_dir, 'loss_curve.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved loss curve to {out}")

    # ──────────────────────────────────────────────────────────────────────
    # Label efficiency
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _le_sample_by_fraction(train_labels, frac, n_classes, rng):
        indices = []
        for c in range(n_classes):
            idx = np.where(train_labels == c)[0]
            if len(idx) == 0:
                continue
            n_take = min(max(1, int(round(len(idx) * frac))), len(idx))
            indices.append(rng.choice(idx, n_take, replace=False))
        return np.concatenate(indices)

    @staticmethod
    def _le_sample_by_n(train_labels, n, n_classes, rng):
        indices = []
        for c in range(n_classes):
            idx = np.where(train_labels == c)[0]
            if len(idx) == 0:
                continue
            indices.append(rng.choice(idx, min(n, len(idx)), replace=False))
        return np.concatenate(indices)

    @staticmethod
    def _le_run_point(label, sample_fn, train_feats, train_labels,
                      val_feats, val_labels, n_classes, n_repeats, epochs, n_jobs):
        bal_accs, mean_aps, ns = [], [], []
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed=42 + rep)
            idx = sample_fn(rng)
            clf = LogisticRegression(
                solver='lbfgs', penalty='l2', max_iter=epochs,
                class_weight='balanced', C=1, n_jobs=n_jobs, tol=1e-6,
            )
            clf.fit(train_feats[idx], train_labels[idx])
            val_pred  = clf.predict(val_feats)
            val_proba = clf.predict_proba(val_feats)
            ba = float(balanced_accuracy_score(val_labels, val_pred))
            val_bin = label_binarize(val_labels, classes=list(range(n_classes)))
            if val_bin.shape[1] == 1:
                val_bin = np.hstack([1 - val_bin, val_bin])
            ap = float(np.mean(average_precision_score(val_bin, val_proba, average=None)))
            bal_accs.append(ba)
            mean_aps.append(ap)
            ns.append(len(idx))
            print(f"    rep {rep+1} (n={len(idx)}): bal_acc={ba:.4f}  mean_ap={ap:.4f}")
        print(f"    => bal_acc {np.mean(bal_accs):.4f}±{np.std(bal_accs):.4f}  "
              f"mean_ap {np.mean(mean_aps):.4f}±{np.std(mean_aps):.4f}")
        return {
            'label':        label,
            'n_labeled':    int(np.mean(ns)),
            'n_repeats':    len(bal_accs),
            'bal_acc_mean': float(np.mean(bal_accs)),
            'bal_acc_std':  float(np.std(bal_accs)),
            'mean_ap_mean': float(np.mean(mean_aps)),
            'mean_ap_std':  float(np.std(mean_aps)),
            'bal_acc_runs': [round(v, 5) for v in bal_accs],
            'mean_ap_runs': [round(v, 5) for v in mean_aps],
        }

    @staticmethod
    def _label_efficiency(train_feats, train_labels, val_feats, val_labels,
                          classes, n_classes, work_dir,
                          fractions, n_per_class, n_repeats, epochs, n_jobs,
                          full_lp_point=None):
        import os
        from matplotlib.lines import Line2D

        print("\n=== 8. Label Efficiency ===")
        points = []

        for frac in sorted(set(fractions)):
            reps  = 1 if frac >= 1.0 else n_repeats
            label = f"{frac*100:.0f}%"
            print(f"\n  [{label}] per-class fraction, {reps} repeat(s):")
            p = EvaluateModelRich._le_run_point(
                label=label,
                sample_fn=lambda rng, f=frac: EvaluateModelRich._le_sample_by_fraction(
                    train_labels, f, n_classes, rng),
                train_feats=train_feats, train_labels=train_labels,
                val_feats=val_feats,     val_labels=val_labels,
                n_classes=n_classes, n_repeats=reps, epochs=epochs, n_jobs=n_jobs,
            )
            points.append(p)

        for n in sorted(set(n_per_class)):
            label = f"{n}/cls"
            print(f"\n  [{label}] fixed {n} cells per class, {n_repeats} repeat(s):")
            p = EvaluateModelRich._le_run_point(
                label=label,
                sample_fn=lambda rng, n_=n: EvaluateModelRich._le_sample_by_n(
                    train_labels, n_, n_classes, rng),
                train_feats=train_feats, train_labels=train_labels,
                val_feats=val_feats,     val_labels=val_labels,
                n_classes=n_classes, n_repeats=n_repeats, epochs=epochs, n_jobs=n_jobs,
            )
            points.append(p)

        if full_lp_point is not None:
            points.append(full_lp_point)

        points.sort(key=lambda p: p['n_labeled'])

        with open(os.path.join(work_dir, 'label_efficiency.json'), 'w') as f:
            json.dump({
                'classes': classes, 'n_classes': n_classes,
                'epochs': epochs,   'n_repeats': n_repeats,
                'points': points,
            }, f, indent=2)

        # plot
        x    = np.array([p['n_labeled']    for p in points])
        ba_m = np.array([p['bal_acc_mean'] for p in points])
        ba_s = np.array([p['bal_acc_std']  for p in points])
        ap_m = np.array([p['mean_ap_mean'] for p in points])
        ap_s = np.array([p['mean_ap_std']  for p in points])
        lbls    = [p['label'] for p in points]
        markers = ['s' if p['label'].endswith('/cls') else 'o' for p in points]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'Label Efficiency — {os.path.basename(work_dir)}', fontsize=12)
        for ax, means, stds, ylabel, color in [
            (ax1, ba_m, ba_s, 'Balanced Accuracy (val)', 'steelblue'),
            (ax2, ap_m, ap_s, 'Mean Average Precision (val)', 'darkorange'),
        ]:
            ax.plot(x, means, '-', color=color, linewidth=1.5, zorder=1)
            for xi, yi, si, mk in zip(x, means, stds, markers):
                ax.plot(xi, yi, mk, color=color, markersize=7, zorder=2)
                if si > 0:
                    ax.errorbar(xi, yi, yerr=si, fmt='none', color=color,
                                capsize=3, linewidth=1, zorder=2)
            ax.set_xscale('log')
            ax.set_xlabel('# labeled training cells')
            ax.set_ylabel(ylabel)
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            for lbl, xi, yi in zip(lbls, x, means):
                ax.annotate(lbl, xy=(xi, yi), xytext=(0, 8),
                            textcoords='offset points', ha='center', fontsize=8)
        ax2.legend(handles=[
            Line2D([0], [0], marker='o', color='grey', linestyle='none',
                   markersize=7, label='fraction'),
            Line2D([0], [0], marker='s', color='grey', linestyle='none',
                   markersize=7, label='n/class'),
        ], fontsize=8, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, 'label_efficiency.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved label_efficiency.json + label_efficiency.pdf to {work_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # LISI
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_lisi(features, labels, n_neighbors=90, metric='cosine'):
        """Local Inverse Simpson's Index (Harmony, Korsunsky et al. 2019).

        For each cell, look at its n_neighbors nearest neighbours and compute
        the inverse of Simpson's diversity index over the label distribution.

        Returns a per-cell array of LISI values in [1, N_unique_labels].
        Use with cell-type labels for cLISI (low = compact),
        or with sample IDs for iLISI (high = well mixed).
        """
        n = len(features)
        k = min(n_neighbors, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric,
                                n_jobs=-1, algorithm='auto')
        nbrs.fit(features)
        _, indices = nbrs.kneighbors(features)
        indices = indices[:, 1:]          # exclude self

        unique_labels = np.unique(labels)
        label_to_int  = {l: i for i, l in enumerate(unique_labels)}
        label_ints    = np.array([label_to_int[l] for l in labels])
        n_labels      = len(unique_labels)

        lisi = np.empty(n, dtype=np.float32)
        for i in range(n):
            nbr_lbls = label_ints[indices[i]]
            simpson  = 0.0
            for l in range(n_labels):
                p = np.mean(nbr_lbls == l)
                simpson += p * p
            lisi[i] = 1.0 / simpson if simpson > 0 else 1.0
        return lisi

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

        def _save_metrics():
            with open(f'{work_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        # ── 1. Linear Probe ────────────────────────────────────────────────
        print("\n=== 1. Linear Probe ===")
        clf = LogisticRegression(
            solver='lbfgs', penalty='l2', max_iter=self.epochs,
            class_weight='balanced', C=1, n_jobs=self.n_jobs, verbose=1, tol=1e-6,
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

        # per-class average precision (matches KRONOS paper metric)
        val_labels_bin = label_binarize(val_labels, classes=list(range(n_classes)))
        per_class_ap   = average_precision_score(val_labels_bin, val_proba_lr, average=None)
        mean_ap        = float(np.mean(per_class_ap))
        metrics['linear_probe']['val']['mean_average_precision'] = mean_ap
        metrics['linear_probe']['val']['per_class_ap'] = {
            cls: round(float(per_class_ap[i]), 4) for i, cls in enumerate(classes)
        }
        print(f"  Val mean AP:      {mean_ap:.4f}")
        _save_metrics()

        # ── 2. k-NN ────────────────────────────────────────────────────────
        print(f"\n=== 2. k-NN (k={self.knn_k}, cosine, distance-weighted) ===")
        knn = KNeighborsClassifier(n_neighbors=self.knn_k, metric='cosine',
                                   weights='distance', n_jobs=self.n_jobs)
        knn.fit(train_feats, train_labels)
        val_pred_knn = knn.predict(val_feats)

        metrics['knn'] = {
            'k': self.knn_k,
            'val': {
                'top1_accuracy':          float(accuracy_score(val_labels, val_pred_knn)),
                'top1_balanced_accuracy': float(balanced_accuracy_score(val_labels, val_pred_knn)),
                'f1':                     float(f1_score(val_labels, val_pred_knn, average='weighted')),
            },
        }
        print(f"  Val balanced acc: {metrics['knn']['val']['top1_balanced_accuracy']:.4f}")
        _save_metrics()

        # ── 3. k-means → NMI / ARI ────────────────────────────────────────
        print(f"\n=== 3. k-means Clustering (k={n_classes}) ===")
        all_feats  = np.concatenate([train_feats, val_feats])
        all_labels = np.concatenate([train_labels, val_labels])
        km = MiniBatchKMeans(n_clusters=n_classes, n_init=10, max_iter=300)
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
        _save_metrics()

        # ── 4. Neighbourhood Purity ────────────────────────────────────────
        # Raw purity is biased against rare classes: a class with 1% prevalence
        # has a random-chance purity of 1%, so raw numbers are not comparable
        # across classes. We therefore also report purity_lift = observed /
        # expected-by-chance, which normalises for class frequency.
        print(f"\n=== 4. Neighbourhood Purity (k={self.knn_k}) ===")
        nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1, metric='cosine', n_jobs=self.n_jobs)
        nbrs.fit(val_feats)
        _, nn_idx = nbrs.kneighbors(val_feats)
        # nn_idx[:, 0] is the point itself — skip it
        neighbor_labels = val_labels[nn_idx[:, 1:]]            # (N, k)
        mean_purity = float(np.mean(neighbor_labels == val_labels[:, None]))

        # class frequencies in val set (= expected purity by chance)
        class_freq = np.bincount(val_labels, minlength=n_classes) / len(val_labels)

        per_class_purity = {}
        for c, cls in enumerate(classes):
            mask = val_labels == c
            if mask.sum() == 0:
                continue
            observed  = float(np.mean(neighbor_labels[mask] == c))
            expected  = float(class_freq[c])
            lift      = observed / expected if expected > 0 else float('inf')
            per_class_purity[cls] = {
                'purity':          observed,
                'expected_chance': expected,
                'purity_lift':     lift,
                'n_cells':         int(mask.sum()),
            }

        # mean lift (macro-average across classes)
        mean_lift = float(np.mean([v['purity_lift'] for v in per_class_purity.values()]))

        metrics['neighbourhood_purity'] = {
            'k': self.knn_k,
            'mean_purity': mean_purity,
            'mean_lift':   mean_lift,
            'per_class':   per_class_purity,
        }
        print(f"  Mean neighbourhood purity: {mean_purity:.4f}  |  Mean lift: {mean_lift:.2f}x")
        _save_metrics()

        # ── 5. cLISI — cell-type compactness ──────────────────────────────
        # cLISI ∈ [1, N_types]: 1 = every neighbour same type (compact),
        # N_types = perfectly mixed (no structure).
        # Normalised cLISI = (cLISI - 1) / (N_types - 1) ∈ [0, 1], lower = better.
        # k=90 follows Harmony paper standard — large enough to be discriminative.
        lisi_k = min(self.lisi_k, len(val_feats) - 1)
        print(f"\n=== 5. cLISI — cell-type compactness (k={lisi_k}) ===")
        clisi_vals = self._compute_lisi(val_feats, val_labels_str,
                                        n_neighbors=lisi_k, metric='cosine')
        clisi_mean = float(np.mean(clisi_vals))
        clisi_norm = float((clisi_mean - 1) / max(n_classes - 1, 1))
        per_class_clisi = {
            cls: float(np.mean(clisi_vals[val_labels == c]))
            for c, cls in enumerate(classes)
            if (val_labels == c).sum() > 0
        }
        metrics['clisi'] = {
            'k':           lisi_k,
            'mean':        clisi_mean,
            'normalised':  clisi_norm,
            'note':        'normalised: 0=compact, 1=random; lower is better',
            'per_class':   {k: round(v, 4) for k, v in per_class_clisi.items()},
        }
        print(f"  cLISI (mean): {clisi_mean:.4f}  |  normalised: {clisi_norm:.4f}")
        _save_metrics()

        # ── 6. iLISI — sample integration ─────────────────────────────────
        # iLISI ∈ [1, N_samples]: 1 = all neighbours same sample (not mixed),
        # N_samples = perfectly mixed.
        # Normalised iLISI = (iLISI - 1) / (N_samples - 1) ∈ [0, 1], higher = better.
        print(f"\n=== 6. iLISI — sample integration (k={lisi_k}) ===")
        n_unique_samples = len(np.unique(val_ids))
        ilisi_vals = self._compute_lisi(val_feats, val_ids,
                                        n_neighbors=lisi_k, metric='cosine')
        ilisi_mean = float(np.mean(ilisi_vals))
        ilisi_norm = float((ilisi_mean - 1) / max(n_unique_samples - 1, 1))
        metrics['ilisi'] = {
            'k':              lisi_k,
            'mean':           ilisi_mean,
            'normalised':     ilisi_norm,
            'n_unique_samples': n_unique_samples,
            'note':           'normalised: 0=separated, 1=fully mixed; higher is better',
        }
        print(f"  iLISI (mean): {ilisi_mean:.4f}  |  normalised: {ilisi_norm:.4f}")
        _save_metrics()

        # keep silhouette-based integration for backwards compat
        sample_integration = float('nan')

        # ── 6. Confusion Matrix (linear probe) ────────────────────────────
        val_pred_str_lr = le.inverse_transform(val_pred_lr)
        val_cm = confusion_matrix(val_labels_str, val_pred_str_lr,
                                  labels=classes, normalize='true')
        fig, ax = plt.subplots(figsize=(max(8, n_classes), max(7, n_classes - 1)))
        ConfusionMatrixDisplay(val_cm, display_labels=classes).plot(
            ax=ax, cmap='Blues', values_format='.2f', xticks_rotation=45)
        ax.set_title('Confusion Matrix — Linear Probe (Val)')
        plt.tight_layout()
        plt.savefig(f'{work_dir}/confusion_matrix.pdf', bbox_inches='tight')
        plt.close()

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

        # ── 7. UMAP ───────────────────────────────────────────────────────
        if HAS_UMAP:
            print("\n=== 7. UMAP ===")
            reducer = umap_lib.UMAP(
                n_components=2, metric='cosine',
                n_neighbors=15, min_dist=0.1,
                n_jobs=self.n_jobs, verbose=True,
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
            plt.savefig(f'{work_dir}/umap.pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved UMAP to {work_dir}/umap.pdf")

            # UMAP coloured by sample ID
            unique_ids = np.unique(val_ids)
            n_samples_ids = len(unique_ids)
            id_to_int = {sid: i for i, sid in enumerate(unique_ids)}
            val_id_ints = np.array([id_to_int[sid] for sid in val_ids])
            cmap_s = plt.get_cmap('nipy_spectral', n_samples_ids)
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            for i, sid in enumerate(unique_ids):
                mask = val_id_ints == i
                ax2.scatter(val_emb[mask, 0], val_emb[mask, 1],
                            s=1, alpha=0.4, color=cmap_s(i), label=str(sid), rasterized=True)
            if n_samples_ids <= 30:
                ax2.legend(markerscale=6, bbox_to_anchor=(1.02, 1),
                           loc='upper left', fontsize=8, frameon=False, title='Sample ID')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.set_title('Val features — UMAP (coloured by sample ID)')
            plt.tight_layout()
            plt.savefig(f'{work_dir}/umap_sample.pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved sample UMAP to {work_dir}/umap_sample.pdf")

            np.savez_compressed(f'{work_dir}/umap_embeddings.npz',
                                embedding=val_emb,
                                labels_num=val_labels,
                                labels_str=val_labels_str,
                                sample_ids=val_ids)
            metrics['umap'] = {'saved': True, 'n_neighbors': 15,
                               'min_dist': 0.1, 'metric': 'cosine'}
        else:
            print("\nSkipping UMAP — install with: pip install umap-learn")
            metrics['umap'] = {'saved': False, 'reason': 'umap-learn not installed'}
        _save_metrics()

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
            top1_pred_lr=train_pred_lr,
            classes=le.classes_,
        )

        # ── Loss curve ────────────────────────────────────────────────────
        self._plot_loss_curve(work_dir)

        # ── Summary print ──────────────────────────────────────────────────
        lp  = metrics['linear_probe']['val']
        knn = metrics['knn']['val']
        cl  = metrics['clustering']['val']
        print(f"""
╔═══════════════════════════════════════════════════════╗
║               Evaluation Summary (Val)               ║
╠═══════════════════════════════════════════════════════╣
║  Linear Probe  bal-acc  {lp['top1_balanced_accuracy']:.4f}   F1 {lp['f1']:.4f}     ║
║  k-NN (k={self.knn_k:2d})    bal-acc  {knn['top1_balanced_accuracy']:.4f}   F1 {knn['f1']:.4f}     ║
║  Clustering    NMI      {cl['nmi']:.4f}   ARI {cl['ari']:.4f}    ║
║  Nbhd purity   raw {mean_purity:.4f}   lift {mean_lift:.2f}x           ║
║  cLISI (norm)  {clisi_norm:.4f}  (0=compact, 1=random)       ║
║  iLISI (norm)  {ilisi_norm:.4f}  (0=separated, 1=mixed)      ║
╚═══════════════════════════════════════════════════════╝
""")
        print(f"Saved metrics to {work_dir}/metrics.json")
