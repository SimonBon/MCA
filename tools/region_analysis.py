#!/usr/bin/env python
"""
Spatial region analysis for a single model on a MCI dataset.

Steps
-----
1. Extract sliding-window patches from the HDF5 file
2. Embed with the frozen backbone (or reload from cache)
3. PCA → k-means at a fixed k
4. Visualise: spatial cluster maps, marker-signal overlay, UMAP, enrichment heatmap

Saved outputs
-------------
  embeddings.npz          — patch embeddings + metadata  (reusable)
  cluster_labels.npy      — cluster assignment per patch  (reusable)
  cluster_composition.json
  spatial_map.png         — cluster map + marker images per patient
  umap.png
  enrichment_heatmap.png
  summary.json

Usage
-----
python tools/region_analysis.py \
    --model_dir  z_RUNS/paper/CODEX_cHL/CIM \
    --h5         /path/to/CODEX_cHL.h5 \
    --markers    /path/to/used_markers.txt \
    --out        z_RUNS/region_analysis/CODEX_cHL_CIM \
    --patch_size 64 \
    --k          6 \
    --display_markers Pan-Keratin CD3 CD68 Vimentin DAPI-01
"""

import argparse, json, os, sys, warnings, shutil
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ── make both `import src.xxx` and `import MCA.src.xxx` work ──────────────
_TOOLS = Path(__file__).resolve().parent
_MCA   = _TOOLS.parent          # …/MCA
_SRC   = _MCA.parent            # …/src  (parent of the MCA package)
for _p in [str(_MCA), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--markers",    required=True, help="used_markers.txt")
    p.add_argument("--out",        required=True)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--k",          type=int, default=6, help="Number of spatial clusters")
    p.add_argument("--pca",        type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_jobs",     type=int, default=8)
    p.add_argument("--n_show",     type=int, default=4, help="Patients to show in spatial map")
    p.add_argument("--gpu",        default="cuda:0")
    p.add_argument("--umap_max",   type=int, default=40_000)
    p.add_argument("--ignore_types", nargs="*", default=["Unidentified"])
    p.add_argument("--display_markers", nargs="+",
                   default=["Pan-Keratin", "CD3", "CD68", "Vimentin", "dsDNA"],
                   help="Marker names for the signal overlay column")
    p.add_argument("--reload", action="store_true",
                   help="Re-embed even if embeddings.npz already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_backbone(model_dir, device):
    """Load backbone from a run dir (with or without timestamp subfolders)."""
    from src.utils import load_checkpoint

    model_dir = Path(model_dir)
    ts_dirs   = [p for p in model_dir.iterdir()
                 if p.is_dir() and p.name[:4].isdigit()]
    if not ts_dirs:
        # Paper-style dir: copy config into expected subfolder layout
        cfg_files = list(model_dir.glob("*.py"))
        if not cfg_files:
            raise FileNotFoundError(f"No config .py in {model_dir}")
        vis_dir = model_dir / "20260101_000000" / "vis_data"
        vis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_files[0], vis_dir / "config.py")

    result   = load_checkpoint(str(model_dir), device=device)
    backbone = result["model"].backbone
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def decode(x):
    return x.decode() if isinstance(x, bytes) else x


def load_dataset(h5_path, markers_path, ignore_types):
    import h5py, numpy as np
    with h5py.File(h5_path, "r") as f:
        all_marker_names = [decode(m) for m in f["marker_names"][:]]
        cell_sids        = [decode(s) for s in f["coords"]["sample_id"][:]]
        cell_y           = f["coords"]["DIM1"][:].astype(int)
        cell_x           = f["coords"]["DIM2"][:].astype(int)
        cell_types       = [decode(a) for a in f["annotation"][:]]
        unique_sids      = [decode(s) for s in f["sample_ids"][:]]

    with open(markers_path) as fh:
        used = [l.strip() for l in fh if l.strip()]
    m2i            = {m: i for i, m in enumerate(all_marker_names)}
    marker_indices = np.array([m2i[m] for m in used])

    ignore         = set(ignore_types or [])
    cell_mask      = np.array([a not in ignore for a in cell_types])
    return dict(
        all_marker_names = all_marker_names,
        cell_sids        = np.array(cell_sids),
        cell_y           = cell_y,
        cell_x           = cell_x,
        cell_types       = np.array(cell_types),
        unique_sids      = unique_sids,
        marker_indices   = marker_indices,
        cell_mask        = cell_mask,
    )


def build_grid(h5_path, unique_sids, ps, stride):
    import h5py
    meta = []
    with h5py.File(h5_path, "r") as f:
        for sid in unique_sids:
            H, W = f["data"][sid]["image"].shape[:2]
            for y in range(0, H - ps + 1, stride):
                for x in range(0, W - ps + 1, stride):
                    meta.append((sid, y, x, H, W))
    return meta


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

import torch

@torch.no_grad()
def _flush(patches, idx, emb, backbone, device):
    import numpy as np, torch.nn.functional as F
    t = torch.from_numpy(np.array(patches, dtype=np.float32)).to(device)
    f = backbone(t)
    if isinstance(f, (tuple, list)):
        f = f[0]
    f = F.normalize(f.squeeze(-1).squeeze(-1), dim=1).cpu().numpy().astype(np.float16)
    for k, i in enumerate(idx):
        emb[i] = f[k]


def embed_patches(backbone, h5_path, meta, marker_indices,
                  sid2idx, unique_sids, ps, feat_dim, batch_size):
    import numpy as np, h5py
    from tqdm import tqdm
    device = next(backbone.parameters()).device
    emb    = np.zeros((len(meta), feat_dim), dtype=np.float16)
    with h5py.File(h5_path, "r") as f:
        bar = tqdm(total=len(meta), desc="Embedding patches", unit="patch", dynamic_ncols=True)
        for sid in unique_sids:
            img  = f["data"][sid]["image"][:].astype(np.float32)[:, :, marker_indices]
            buf_p, buf_i = [], []
            for i in sid2idx[sid]:
                _, y, x, _, _ = meta[i]
                buf_p.append(img[y:y+ps, x:x+ps].transpose(2, 0, 1))
                buf_i.append(i)
                if len(buf_p) == batch_size:
                    _flush(buf_p, buf_i, emb, backbone, device)
                    bar.update(len(buf_p))
                    buf_p, buf_i = [], []
            if buf_p:
                _flush(buf_p, buf_i, emb, backbone, device)
                bar.update(len(buf_p))
        bar.close()
    return emb


# ---------------------------------------------------------------------------
# Cell → patch mapping
# ---------------------------------------------------------------------------

def map_cells(cell_mask, cell_sids, cell_y, cell_x, cell_types,
              sample_dims, patch_index, ps, stride):
    from tqdm import tqdm
    p2ann = defaultdict(list)
    for i in tqdm(range(len(cell_sids)), desc="Mapping cells", unit="cell", dynamic_ncols=True):
        if not cell_mask[i]:
            continue
        sid = cell_sids[i]
        if sid not in sample_dims:
            continue
        cy, cx   = int(cell_y[i]), int(cell_x[i])
        H, W     = sample_dims[sid]
        py0 = (max(0, cy - ps + 1) // stride) * stride
        px0 = (max(0, cx - ps + 1) // stride) * stride
        for py in range(py0, min(cy + 1, H - ps + 1), stride):
            for px in range(px0, min(cx + 1, W - ps + 1), stride):
                pidx = patch_index.get((sid, py, px))
                if pidx is not None:
                    p2ann[pidx].append(cell_types[i])
    return p2ann


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def percentile_norm(img, lo=1, hi=99):
    import numpy as np
    lo_v, hi_v = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)


def plot_spatial_and_markers(labels, k, show_sids, meta, sid2idx,
                              sample_dims, ps, h5_path,
                              all_marker_names, display_markers, save_path):
    """One row per patient: cluster map | marker1 | marker2 | ..."""
    import h5py, numpy as np
    import matplotlib.pyplot as plt, matplotlib.patches as mpatches

    amn_str   = [m.decode() if isinstance(m, bytes) else m for m in all_marker_names]
    available = [m for m in display_markers if m in amn_str]
    d_idx     = [amn_str.index(m) for m in available]

    cmap   = plt.cm.get_cmap("tab10", k)
    n_rows = len(show_sids)
    n_cols = 1 + len(available)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    with h5py.File(h5_path, "r") as f:
        for row, sid in enumerate(show_sids):
            H, W     = sample_dims[sid]
            img_full = f["data"][sid]["image"][:].astype(np.float32)

            # Cluster map
            canvas = np.full((H, W, 4), [0.12, 0.12, 0.12, 1.0])
            for i in sid2idx[sid]:
                _, y, x, _, _ = meta[i]
                canvas[y:y+ps, x:x+ps] = cmap(int(labels[i]))
            axes[row, 0].imshow(canvas)
            axes[row, 0].set_title(f"Clusters · {sid}", fontsize=8, fontweight="bold")
            axes[row, 0].axis("off")

            # Marker channels
            for col, (midx, mname) in enumerate(zip(d_idx, available), start=1):
                axes[row, col].imshow(percentile_norm(img_full[:, :, midx]), cmap="inferno")
                axes[row, col].set_title(mname, fontsize=8)
                axes[row, col].axis("off")

    handles = [mpatches.Patch(color=cmap(c), label=f"C{c}") for c in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=k, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_umap(emb_pca, labels, k, n_jobs, umap_max, save_path):
    import numpy as np, matplotlib.pyplot as plt
    import umap as umap_lib
    n    = min(umap_max, len(emb_pca))
    idx  = np.random.RandomState(42).choice(len(emb_pca), n, replace=False)
    xy   = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         metric="cosine", n_jobs=n_jobs, verbose=False
                         ).fit_transform(emb_pca[idx].astype(np.float32))
    cmap = plt.cm.get_cmap("tab10", k)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels[idx], cmap=cmap,
                    vmin=-0.5, vmax=k - 0.5, s=2, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Cluster", ticks=range(k))
    ax.set(title=f"UMAP  k={k}", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_enrichment(comp, labels, global_freq, all_ct, k, save_path,
                    vmin=-2.5, vmax=2.5, thresh=0.5):
    import numpy as np, matplotlib.pyplot as plt

    eps  = 1e-6
    fc   = np.clip(np.log2((comp + eps) / (global_freq[None] + eps)), vmin, vmax)
    rng  = fc.max(0) - fc.min(0)
    keep = np.where(rng > thresh)[0]
    if len(keep) == 0:
        keep = np.argsort(rng)[::-1][:10]

    ct_labels   = [all_ct[i] for i in keep]
    sizes       = np.bincount(labels, minlength=k)

    def top2(row):
        t = row.argsort()[::-1][:2]
        return " / ".join(all_ct[i].replace("_", " ") for i in t)

    row_labels = [f"C{i} ({sizes[i]/len(labels)*100:.0f}%)  {top2(fc[i])}"
                  for i in range(k)]

    fig, ax = plt.subplots(figsize=(max(10, len(keep) * 0.75), 1.2 + k * 0.85))
    im = ax.imshow(fc[:, keep], aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(ct_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="log2 fold enrichment", shrink=0.7)
    for ci in range(k):
        for cj, orig in enumerate(keep):
            v = fc[ci, orig]
            if abs(v) >= 1.0:
                ax.text(cj, ci, f"{v:+.1f}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="white" if abs(v) > 1.8 else "black")
    ax.set_title("Cell-type enrichment per cluster  "
                 "(log2 cluster fraction / global fraction)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    import matplotlib; matplotlib.use("Agg")

    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(args.n_jobs)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(args.n_jobs)
    except ImportError:
        pass

    DEVICE = args.gpu if torch.cuda.is_available() else "cpu"
    ps     = args.patch_size
    stride = ps // 2
    k      = args.k
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}   Patch: {ps}px   k: {k}")
    print(f"Output : {out}")

    # Load dataset metadata
    ds = load_dataset(args.h5, args.markers, args.ignore_types)
    unique_sids    = ds["unique_sids"]
    marker_indices = ds["marker_indices"]

    # Build patch grid
    meta   = build_grid(args.h5, unique_sids, ps, stride)
    pid    = {(s, y, x): i for i, (s, y, x, H, W) in enumerate(meta)}
    sid2idx = defaultdict(list)
    for i, (sid, *_) in enumerate(meta):
        sid2idx[sid].append(i)
    sample_dims = {sid: (H, W) for sid, y, x, H, W in meta}
    print(f"Patches: {len(meta):,}   Patients: {len(unique_sids)}")

    # Embeddings (load from cache or compute)
    emb_path = out / "embeddings.npz"
    if not args.reload and emb_path.exists():
        print(f"Loading cached embeddings from {emb_path}")
        cache = np.load(emb_path, allow_pickle=True)
        emb   = cache["embeddings"]
    else:
        backbone = load_backbone(args.model_dir, DEVICE)
        with torch.no_grad():
            dummy    = torch.zeros(1, len(marker_indices), ps, ps, device=DEVICE)
            feat_dim = backbone(dummy)[0].squeeze(-1).squeeze(-1).shape[1]
        emb = embed_patches(backbone, args.h5, meta, marker_indices,
                            sid2idx, unique_sids, ps, feat_dim, args.batch_size)
        del backbone; torch.cuda.empty_cache()

        # Save patch metadata alongside embeddings for later reuse
        sids_arr = np.array([m[0] for m in meta])
        ys_arr   = np.array([m[1] for m in meta])
        xs_arr   = np.array([m[2] for m in meta])
        np.savez_compressed(emb_path,
                            embeddings=emb,
                            patch_sids=sids_arr,
                            patch_y=ys_arr,
                            patch_x=xs_arr)
        print(f"Saved embeddings → {emb_path}  ({emb.nbytes/1024**2:.0f} MB)")

    # PCA + clustering
    print("PCA + k-means...")
    pca     = PCA(n_components=args.pca, random_state=42)
    emb_pca = pca.fit_transform(emb.astype(np.float32))
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    km     = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10,
                             batch_size=4096, max_iter=300)
    labels = km.fit_predict(emb_pca)
    np.save(out / "cluster_labels.npy", labels)
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")

    # Cell → patch mapping
    p2ann = map_cells(ds["cell_mask"], ds["cell_sids"], ds["cell_y"], ds["cell_x"],
                      ds["cell_types"], sample_dims, pid, ps, stride)
    all_ct = sorted({a for anns in p2ann.values() for a in anns})

    # Cluster composition
    from collections import Counter as _Counter, defaultdict as _dd
    cl_counts = _dd(_Counter)
    for pidx, anns in p2ann.items():
        cl_counts[int(labels[pidx])].update(anns)
    comp = np.zeros((k, len(all_ct)))
    for cl in range(k):
        t = sum(cl_counts[cl].values())
        if t:
            comp[cl] = [cl_counts[cl].get(ct, 0) / t for ct in all_ct]

    global_counts = _Counter()
    for anns in p2ann.values():
        global_counts.update(anns)
    total       = sum(global_counts.values())
    global_freq = np.array([global_counts.get(ct, 0) / total for ct in all_ct])

    with open(out / "cluster_composition.json", "w") as fh:
        json.dump({
            "cell_types": all_ct,
            "global_freq": global_freq.tolist(),
            "clusters": {
                str(cl): {"composition": comp[cl].tolist(),
                           "n_patches": int((labels == cl).sum())}
                for cl in range(k)
            }
        }, fh, indent=2)

    # Visualisations
    show_sids = [s for s, _ in Counter(m[0] for m in meta).most_common(args.n_show)]

    plot_spatial_and_markers(
        labels, k, show_sids, meta, sid2idx, sample_dims, ps,
        args.h5, ds["all_marker_names"], args.display_markers,
        out / "spatial_map.png",
    )
    plot_umap(emb_pca, labels, k, args.n_jobs, args.umap_max, out / "umap.png")
    plot_enrichment(comp, labels, global_freq, all_ct, k, out / "enrichment_heatmap.png")

    # Summary
    from scipy.spatial.distance import jensenshannon
    js_vals = []
    for i in range(k):
        for j in range(i+1, k):
            p = comp[i]+1e-9; p /= p.sum()
            q = comp[j]+1e-9; q /= q.sum()
            js_vals.append(float(jensenshannon(p, q)))

    summary = {
        "model_dir":            str(args.model_dir),
        "dataset":              str(args.h5),
        "patch_size":           ps,
        "k":                    k,
        "n_patches":            len(meta),
        "n_patients":           len(unique_sids),
        "pca_variance":         float(pca.explained_variance_ratio_.sum()),
        "between_cluster_js":   float(np.mean(js_vals)),
        "cluster_sizes":        np.bincount(labels).tolist(),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nDone — results in {out}")


if __name__ == "__main__":
    main()
