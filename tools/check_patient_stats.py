"""
Benchmark per-patient stat estimation strategies for C_PatientStyleTransfer.

For a single patient, computes "ground truth" mean/std from the full image,
then measures how quickly different sampling strategies converge to it,
and how long each takes.

Usage:
    python tools/check_patient_stats.py --h5 /path/to/dataset.h5 \
        --markers /path/to/used_markers.txt [--patient_id PATIENT_ID]
"""

import argparse
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h5',      required=True)
    p.add_argument('--markers', required=True)
    p.add_argument('--patient_id', default=None,
                   help='Patient ID to benchmark (default: first patient)')
    p.add_argument('--out', default='patient_stats_benchmark.png')
    return p.parse_args()


def load_marker_indices(h5f, markers_path):
    all_markers = h5f['marker_names'][:].astype(str)
    markers     = np.loadtxt(markers_path, dtype=str, delimiter=',')
    m2i         = {name: i for i, name in enumerate(all_markers)}
    used_idx    = np.array(sorted([m2i[m] for m in markers]))
    return used_idx, all_markers[used_idx]


def ground_truth(img_ds, used_idx):
    """Read full image — slow but exact."""
    t0   = time.perf_counter()
    full = img_ds[:, :, used_idx].astype(np.float32)
    flat = full.reshape(-1, len(used_idx))
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    elapsed = time.perf_counter() - t0
    return mean, std, elapsed, len(flat)


def strip_stats(img_ds, used_idx, n_pixels):
    """Read a contiguous horizontal strip from the top (current approach)."""
    H, W = img_ds.shape[:2]
    n_rows = max(1, int(np.ceil(n_pixels / W)))
    n_rows = min(n_rows, H)
    t0     = time.perf_counter()
    chunk  = img_ds[:n_rows, :, used_idx].astype(np.float32)
    flat   = chunk.reshape(-1, len(used_idx))
    if len(flat) > n_pixels:
        flat = flat[:n_pixels]
    mean    = flat.mean(axis=0)
    std     = flat.std(axis=0)
    elapsed = time.perf_counter() - t0
    return mean, std, elapsed, len(flat)


def random_row_stats(img_ds, used_idx, n_pixels):
    """Read n randomly chosen complete rows (contiguous per row, but random row indices)."""
    H, W    = img_ds.shape[:2]
    n_rows  = max(1, int(np.ceil(n_pixels / W)))
    n_rows  = min(n_rows, H)
    row_idx = np.sort(np.random.choice(H, n_rows, replace=False))
    t0      = time.perf_counter()
    chunks  = [img_ds[r:r+1, :, used_idx].astype(np.float32).reshape(-1, len(used_idx))
               for r in row_idx]
    flat    = np.concatenate(chunks, axis=0)
    if len(flat) > n_pixels:
        flat = flat[:n_pixels]
    mean    = flat.mean(axis=0)
    std     = flat.std(axis=0)
    elapsed = time.perf_counter() - t0
    return mean, std, elapsed, len(flat)


def relative_error(est, truth):
    """Mean relative error across channels (ignoring near-zero truth values)."""
    mask = np.abs(truth) > 1e-4
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(est[mask] - truth[mask]) / np.abs(truth[mask])))


def main():
    args = parse_args()

    with h5py.File(args.h5, 'r') as f:
        used_idx, used_names = load_marker_indices(f, args.markers)

        patients = list(f['data'].keys())
        pid = args.patient_id or patients[0]
        assert pid in f['data'], f"Patient {pid} not found. Available: {patients[:5]}"

        img_ds = f['data'][pid]['image']
        H, W, C = img_ds.shape
        total_pixels = H * W
        print(f"\nPatient: {pid}")
        print(f"Image shape: {H} x {W} x {C}  ({total_pixels:,} total pixels)")
        print(f"Using {len(used_idx)} markers")

        # Ground truth
        print("\nComputing ground truth (full image)...")
        gt_mean, gt_std, gt_time, gt_n = ground_truth(img_ds, used_idx)
        print(f"  Full image: {gt_n:,} pixels in {gt_time:.2f}s")

        # How much error is actually tolerable?
        # Style transfer: x_new = (x - src_mean) / src_std * tgt_std + tgt_mean
        # If mean is off by δ, the output shifts by δ — same magnitude as the
        # mean itself. A 5% relative error on mean is probably fine for augmentation.
        print(f"\n  GT mean (first 5 markers): {gt_mean[:5].round(4)}")
        print(f"  GT std  (first 5 markers): {gt_std[:5].round(4)}")

        # Benchmark different sample sizes
        sample_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
        sample_sizes = [s for s in sample_sizes if s < total_pixels]

        results_strip = []
        results_rows  = []

        print(f"\n{'Strategy':<20} {'N pixels':>10} {'Time (s)':>10} {'Mean err%':>12} {'Std err%':>12}")
        print('-' * 70)

        for n in sample_sizes:
            # Strip
            m, s, t, actual_n = strip_stats(img_ds, used_idx, n)
            me = relative_error(m, gt_mean) * 100
            se = relative_error(s, gt_std)  * 100
            results_strip.append((actual_n, t, me, se))
            print(f"{'strip':<20} {actual_n:>10,} {t:>10.3f} {me:>11.2f}% {se:>11.2f}%")

            # Random rows
            m, s, t, actual_n = random_row_stats(img_ds, used_idx, n)
            me = relative_error(m, gt_mean) * 100
            se = relative_error(s, gt_std)  * 100
            results_rows.append((actual_n, t, me, se))
            print(f"{'random rows':<20} {actual_n:>10,} {t:>10.3f} {me:>11.2f}% {se:>11.2f}%")

        print(f"\n{'Full image':<20} {gt_n:>10,} {gt_time:>10.2f} {'0.00%':>12} {'0.00%':>12}")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'Per-patient stat estimation — patient {pid}', fontsize=11)

        for ax, col, ylabel in [
            (axes[0], 2, 'Mean relative error (%)'),
            (axes[1], 3, 'Std relative error (%)'),
            (axes[2], 1, 'Time (s)'),
        ]:
            ns_strip = [r[0] for r in results_strip]
            ns_rows  = [r[0] for r in results_rows]
            ax.plot(ns_strip, [r[col] for r in results_strip], 'o-', label='strip (top)')
            ax.plot(ns_rows,  [r[col] for r in results_rows],  's-', label='random rows')
            ax.axvline(total_pixels, color='grey', linestyle=':', alpha=0.5, label='full image')
            if col in (2, 3):
                ax.axhline(5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
            ax.set_xscale('log')
            ax.set_xlabel('# pixels sampled')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()
        plt.savefig(args.out, dpi=130, bbox_inches='tight')
        print(f"\nSaved plot to {args.out}")


if __name__ == '__main__':
    main()
