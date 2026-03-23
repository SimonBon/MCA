#!/usr/bin/env python3
"""LP benchmark: compare multiple classification heads on frozen features.

Methods: sklearn (L2), L1, PCA+LP, MLP, concat (two feature sets)

Usage:
    # L2 LP with C grid
    python tools/benchmark_lp.py --method sklearn \
        --train z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/train_results.npz \
        --val   z_RUNS/paper_clean/CODEX_cHL_KRONOS18/CIM_Funnel_Large/val_results.npz \
        --c_grid 1e-4 1e-3 1e-2 0.1 1 10 100

    # PCA + LP
    python tools/benchmark_lp.py --method pca_lp --train ... --val ...

    # L1 LP
    python tools/benchmark_lp.py --method l1 --train ... --val ...

    # MLP head
    python tools/benchmark_lp.py --method mlp --train ... --val ...

    # Concat two feature sets + LP
    python tools/benchmark_lp.py --method concat \
        --train  .../CIM_Funnel_Large/train_results.npz \
        --val    .../CIM_Funnel_Large/val_results.npz \
        --train2 .../ExprBaseline/train_results.npz \
        --val2   .../ExprBaseline/val_results.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.preprocessing import label_binarize


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_proba, classes):
    n = len(classes)
    bal  = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ybin = label_binarize(y_true, classes=list(range(n)))
    if ybin.shape[1] == 1:
        ybin = np.hstack([1 - ybin, ybin])
    ap_per_class = average_precision_score(ybin, y_proba, average=None)
    mAP  = float(np.mean(ap_per_class))
    per_class_ap = {cls: round(float(ap_per_class[i]), 4) for i, cls in enumerate(classes)}
    return bal, f1, mAP, per_class_ap


def load_npz(train_path, val_path):
    tr = np.load(train_path, allow_pickle=True)
    vl = np.load(val_path,   allow_pickle=True)
    X_train = tr['features'].astype(np.float32)
    X_val   = vl['features'].astype(np.float32)

    if 'labels_str' in tr.files:
        le = LabelEncoder().fit(
            np.concatenate([tr['labels_str'], vl['labels_str']]))
        y_train = le.transform(tr['labels_str'])
        y_val   = le.transform(vl['labels_str'])
        classes = list(le.classes_)
    else:
        y_train = tr['labels_num'].astype(int)
        y_val   = vl['labels_num'].astype(int)
        classes = list(tr['classes']) if 'classes' in tr.files else list(range(int(y_train.max()) + 1))

    return X_train, X_val, y_train.astype(np.int32), y_val.astype(np.int32), classes


def make_synthetic(n_train, n_val, n_dims, n_classes):
    rng = np.random.default_rng(42)
    means = rng.standard_normal((n_classes, n_dims)).astype(np.float32) * 0.5
    y_train = rng.integers(0, n_classes, n_train).astype(np.int32)
    y_val   = rng.integers(0, n_classes, n_val).astype(np.int32)
    X_train = rng.standard_normal((n_train, n_dims)).astype(np.float32) + means[y_train]
    X_val   = rng.standard_normal((n_val,   n_dims)).astype(np.float32) + means[y_val]
    classes = [str(i) for i in range(n_classes)]
    return X_train, X_val, y_train, y_val, classes


def print_summary(results):
    print('\n' + '='*65)
    print(f'{"Method":<18} {"Time(s)":>8} {"Bal Acc":>9} {"F1":>8} {"mAP":>8}')
    print('-'*65)
    for name, r in results.items():
        print(f'{name:<18} {r["time"]:>8.1f} {r["bal"]:>9.4f} '
              f'{r["f1"]:>8.4f} {r["mAP"]:>8.4f}')
    print('='*65)


def save_results(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_path}', flush=True)


# ── sklearn L2 LP ─────────────────────────────────────────────────────────────

def run_sklearn(X_train, X_val, y_train, y_val, classes, max_iter, n_jobs,
                c_grid=None, solver='lbfgs', tol=1e-6):
    grid = c_grid if c_grid is not None else [1.0]
    print(f'\n[sklearn L2]  {solver}  tol={tol}  max_iter={max_iter}  '
          f'C={[f"{c:.0e}" for c in grid]}', flush=True)

    t0 = time.time()
    best_C, best_score, best_clf = None, -1.0, None
    for C in grid:
        clf = skLR(solver=solver, penalty='l2', C=C, max_iter=max_iter,
                   class_weight='balanced', n_jobs=n_jobs, tol=tol)
        clf.fit(X_train, y_train)
        score = float(balanced_accuracy_score(y_val, clf.predict(X_val)))
        print(f'  C={C:.0e}  bal_acc={score:.4f}', flush=True)
        if score > best_score:
            best_score, best_C, best_clf = score, C, clf
    fit_time = time.time() - t0

    print(f'  best C={best_C:.0e}  search={fit_time:.1f}s', flush=True)
    preds = best_clf.predict(X_val)
    proba = best_clf.predict_proba(X_val)
    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    print(f'  bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}', flush=True)
    return fit_time, bal, f1, mAP, per_class_ap


# ── L1 LP ─────────────────────────────────────────────────────────────────────

def run_l1(X_train, X_val, y_train, y_val, classes, max_iter, c_grid=None):
    grid = c_grid if c_grid is not None else [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    print(f'\n[L1 LP]  liblinear  max_iter={max_iter}  '
          f'C={[f"{c:.0e}" for c in grid]}', flush=True)

    t0 = time.time()
    best_C, best_score, best_clf = None, -1.0, None
    for C in grid:
        clf = skLR(solver='liblinear', penalty='l1', C=C, max_iter=max_iter,
                   class_weight='balanced', tol=1e-6)
        clf.fit(X_train, y_train)
        score = float(balanced_accuracy_score(y_val, clf.predict(X_val)))
        print(f'  C={C:.0e}  bal_acc={score:.4f}', flush=True)
        if score > best_score:
            best_score, best_C, best_clf = score, C, clf
    fit_time = time.time() - t0

    print(f'  best C={best_C:.0e}  time={fit_time:.1f}s', flush=True)
    preds = best_clf.predict(X_val)
    proba = best_clf.predict_proba(X_val)
    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    print(f'  bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}', flush=True)
    return fit_time, bal, f1, mAP, per_class_ap


# ── PCA + LP ──────────────────────────────────────────────────────────────────

def run_pca_lp(X_train, X_val, y_train, y_val, classes, max_iter, n_jobs,
               pca_dims=(64, 128, 256), c_grid=None):
    from sklearn.decomposition import PCA

    grid = c_grid if c_grid is not None else [0.01, 0.1, 1.0]
    print(f'\n[PCA+LP]  pca_dims={list(pca_dims)}  '
          f'C={[f"{c:.0e}" for c in grid]}', flush=True)

    t0 = time.time()
    best_overall, best_dim, best_C_overall = -1.0, None, None
    best_clf_overall, best_Xvl_overall = None, None

    for n_comp in pca_dims:
        if n_comp >= X_train.shape[1]:
            print(f'  skipping PCA n={n_comp} (>= feature dim {X_train.shape[1]})', flush=True)
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        Xtr = pca.fit_transform(X_train)
        Xvl = pca.transform(X_val)
        var = pca.explained_variance_ratio_.sum()
        print(f'  PCA n={n_comp}  var_explained={var:.3f}', flush=True)

        for C in grid:
            clf = skLR(solver='lbfgs', penalty='l2', C=C, max_iter=max_iter,
                       class_weight='balanced', n_jobs=n_jobs, tol=1e-6)
            clf.fit(Xtr, y_train)
            score = float(balanced_accuracy_score(y_val, clf.predict(Xvl)))
            print(f'    C={C:.0e}  bal_acc={score:.4f}', flush=True)
            if score > best_overall:
                best_overall = score
                best_dim, best_C_overall = n_comp, C
                best_clf_overall, best_Xvl_overall = clf, Xvl

    fit_time = time.time() - t0
    print(f'  best: PCA n={best_dim}  C={best_C_overall:.0e}  '
          f'total={fit_time:.1f}s', flush=True)
    preds = best_clf_overall.predict(best_Xvl_overall)
    proba = best_clf_overall.predict_proba(best_Xvl_overall)
    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    print(f'  bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}', flush=True)
    return fit_time, bal, f1, mAP, per_class_ap


# ── MLP head ──────────────────────────────────────────────────────────────────

def run_mlp(X_train, X_val, y_train, y_val, classes,
            hidden_sizes=(256,), epochs=300, lr=1e-3, wd=1e-4,
            batch_size=1024, patience=30, device='cpu'):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    n_in, n_cls = X_train.shape[1], len(classes)
    print(f'\n[MLP]  {n_in}→{"→".join(str(h) for h in hidden_sizes)}→{n_cls}  '
          f'epochs={epochs}  lr={lr}  device={device}', flush=True)

    # class weights for balanced loss
    counts = np.bincount(y_train, minlength=n_cls).astype(np.float32)
    weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
    weights = weights / weights.sum() * n_cls

    # build model
    layers = []
    in_dim = n_in
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
        in_dim = h
    layers.append(nn.Linear(in_dim, n_cls))
    model = nn.Sequential(*layers).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    t0 = time.time()
    best_val_acc, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds_ep = model(X_vl).argmax(1).cpu().numpy()
        val_acc = float(balanced_accuracy_score(y_val, preds_ep))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0:
            print(f'  epoch {epoch}/{epochs}  val_bal_acc={val_acc:.4f}  '
                  f'best={best_val_acc:.4f}', flush=True)

        if no_improve >= patience:
            print(f'  early stop at epoch {epoch}  best={best_val_acc:.4f}', flush=True)
            break

    fit_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_vl)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    print(f'  MLP fit={fit_time:.1f}s  bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}',
          flush=True)
    return fit_time, bal, f1, mAP, per_class_ap


# ── Concat two feature sets + LP ──────────────────────────────────────────────

def run_concat(X_train, X_val, y_train, y_val, classes,
               X_train2, X_val2, max_iter, n_jobs, c_grid=None):
    grid = c_grid if c_grid is not None else [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]

    # scale each block separately on train statistics, then concatenate
    sc1 = StandardScaler().fit(X_train)
    sc2 = StandardScaler().fit(X_train2)
    Xtr = np.concatenate([sc1.transform(X_train),  sc2.transform(X_train2)],  axis=1)
    Xvl = np.concatenate([sc1.transform(X_val),    sc2.transform(X_val2)],    axis=1)

    print(f'\n[Concat LP]  dim1={X_train.shape[1]}  dim2={X_train2.shape[1]}  '
          f'total={Xtr.shape[1]}  C={[f"{c:.0e}" for c in grid]}', flush=True)

    t0 = time.time()
    best_C, best_score, best_clf = None, -1.0, None
    for C in grid:
        clf = skLR(solver='lbfgs', penalty='l2', C=C, max_iter=max_iter,
                   class_weight='balanced', n_jobs=n_jobs, tol=1e-6)
        clf.fit(Xtr, y_train)
        score = float(balanced_accuracy_score(y_val, clf.predict(Xvl)))
        print(f'  C={C:.0e}  bal_acc={score:.4f}', flush=True)
        if score > best_score:
            best_score, best_C, best_clf = score, C, clf
    fit_time = time.time() - t0

    print(f'  best C={best_C:.0e}  time={fit_time:.1f}s', flush=True)
    preds = best_clf.predict(Xvl)
    proba = best_clf.predict_proba(Xvl)
    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    print(f'  bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}', flush=True)
    return fit_time, bal, f1, mAP, per_class_ap


# ── cuML LP ───────────────────────────────────────────────────────────────────

def run_cuml(X_train, X_val, y_train, y_val, classes, n_trials, max_iter,
             c_min=1e-4, c_max=1e-1, c_grid=None):
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLR

    grid = c_grid if c_grid is not None else [
        10 ** e for e in np.linspace(np.log10(c_min), np.log10(c_max), n_trials)
    ]
    print(f'\n[cuML]  C={[f"{c:.0e}" for c in grid]}  max_iter={max_iter}', flush=True)

    X_tr_gpu = cp.asarray(X_train)
    X_vl_gpu = cp.asarray(X_val)
    y_tr_gpu = cp.asarray(y_train)
    t0 = time.time()

    best_C, best_score = None, -1.0
    for C in grid:
        clf = cuLR(C=C, max_iter=max_iter, class_weight='balanced',
                   output_type='numpy', verbose=False, tol=1e-6)
        clf.fit(X_tr_gpu, y_tr_gpu)
        score = float(balanced_accuracy_score(y_val, clf.predict(X_vl_gpu)))
        print(f'  C={C:.0e}  bal_acc={score:.4f}', flush=True)
        if score > best_score:
            best_score, best_C = score, C
    search_time = time.time() - t0

    t1 = time.time()
    clf = cuLR(C=best_C, max_iter=max_iter, class_weight='balanced',
               output_type='numpy', verbose=False)
    clf.fit(X_tr_gpu, y_tr_gpu)
    preds = clf.predict(X_vl_gpu)
    proba = clf.predict_proba(X_vl_gpu)
    fit_time = time.time() - t1

    bal, f1, mAP, per_class_ap = evaluate(y_val, preds, proba, classes)
    total = search_time + fit_time
    print(f'  best C={best_C:.4g}  total={total:.1f}s  '
          f'bal={bal:.4f}  F1={f1:.4f}  mAP={mAP:.4f}', flush=True)
    return total, bal, f1, mAP, per_class_ap


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--train',   default=None, help='Path to train_results.npz')
    p.add_argument('--val',     default=None, help='Path to val_results.npz')
    p.add_argument('--train2',  default=None, help='Second feature set (for concat)')
    p.add_argument('--val2',    default=None, help='Second feature set val (for concat)')
    p.add_argument('--n_train',   type=int, default=100_000)
    p.add_argument('--n_val',     type=int, default=30_000)
    p.add_argument('--n_dims',    type=int, default=6_912)
    p.add_argument('--n_classes', type=int, default=16)
    # Methods
    p.add_argument('--method', nargs='+',
                   choices=['sklearn', 'l1', 'pca_lp', 'mlp', 'concat', 'cuml'],
                   default=['sklearn'],
                   help='Which methods to run (can specify multiple)')
    # LP common
    p.add_argument('--c_grid',      type=float, nargs='+', default=None)
    p.add_argument('--max_iter_sk', type=int, default=5000)
    p.add_argument('--n_jobs',      type=int, default=8)
    p.add_argument('--solver_sk',   default='lbfgs', choices=['lbfgs', 'saga'])
    p.add_argument('--tol_sk',      type=float, default=1e-6)
    # PCA
    p.add_argument('--pca_dims', type=int, nargs='+', default=[64, 128, 256])
    # MLP
    p.add_argument('--mlp_hidden',  type=int, nargs='+', default=[256])
    p.add_argument('--mlp_epochs',  type=int, default=300)
    p.add_argument('--mlp_lr',      type=float, default=1e-3)
    p.add_argument('--mlp_device',  default='cpu')
    # cuML
    p.add_argument('--max_iter_cu', type=int, default=20000)
    p.add_argument('--n_trials',    type=int, default=7)
    p.add_argument('--c_min',       type=float, default=1e-4)
    p.add_argument('--c_max',       type=float, default=1e2)
    # Preprocessing
    p.add_argument('--subsample',       type=int, default=None)
    p.add_argument('--normalize',       action='store_true')
    p.add_argument('--standard_scale',  action='store_true')
    # Output
    p.add_argument('--out', default=None, help='Save results JSON to this path')
    args = p.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.train and args.val:
        print(f'Loading features from {args.train} ...', flush=True)
        X_train, X_val, y_train, y_val, classes = load_npz(args.train, args.val)
    else:
        print(f'Generating synthetic data: {args.n_train}×{args.n_dims} train, '
              f'{args.n_val}×{args.n_dims} val, {args.n_classes} classes', flush=True)
        X_train, X_val, y_train, y_val, classes = make_synthetic(
            args.n_train, args.n_val, args.n_dims, args.n_classes)

    if args.subsample is not None and args.subsample < len(X_train):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), args.subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f'Subsampled train to {args.subsample} cells', flush=True)

    if args.normalize:
        X_train = normalize(X_train, norm='l2')
        X_val   = normalize(X_val,   norm='l2')
        print('L2-normalized features', flush=True)
    elif args.standard_scale:
        scaler  = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        print('StandardScaler applied', flush=True)

    print(f'Data: train={X_train.shape}  val={X_val.shape}  '
          f'n_classes={len(classes)}', flush=True)

    # Load second feature set for concat
    X_train2, X_val2 = None, None
    if 'concat' in args.method:
        if not (args.train2 and args.val2):
            raise ValueError('--train2 and --val2 required for concat method')
        tr2 = np.load(args.train2, allow_pickle=True)
        vl2 = np.load(args.val2,   allow_pickle=True)
        X_train2 = tr2['features'].astype(np.float32)
        X_val2   = vl2['features'].astype(np.float32)
        if args.subsample is not None:
            X_train2 = X_train2[idx]
        print(f'Second features: train={X_train2.shape}  val={X_val2.shape}', flush=True)

    # ── Run methods ───────────────────────────────────────────────────────────
    results = {}
    c_grid = args.c_grid if args.c_grid else [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]

    if 'sklearn' in args.method:
        t, bal, f1, mAP, per_class_ap = run_sklearn(
            X_train, X_val, y_train, y_val, classes,
            args.max_iter_sk, args.n_jobs,
            c_grid=c_grid, solver=args.solver_sk, tol=args.tol_sk)
        results['sklearn'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    if 'l1' in args.method:
        t, bal, f1, mAP, per_class_ap = run_l1(
            X_train, X_val, y_train, y_val, classes,
            args.max_iter_sk, c_grid=c_grid)
        results['l1'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    if 'pca_lp' in args.method:
        t, bal, f1, mAP, per_class_ap = run_pca_lp(
            X_train, X_val, y_train, y_val, classes,
            args.max_iter_sk, args.n_jobs,
            pca_dims=args.pca_dims, c_grid=c_grid)
        results['pca_lp'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    if 'mlp' in args.method:
        t, bal, f1, mAP, per_class_ap = run_mlp(
            X_train, X_val, y_train, y_val, classes,
            hidden_sizes=args.mlp_hidden, epochs=args.mlp_epochs,
            lr=args.mlp_lr, device=args.mlp_device)
        results['mlp'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    if 'concat' in args.method:
        t, bal, f1, mAP, per_class_ap = run_concat(
            X_train, X_val, y_train, y_val, classes,
            X_train2, X_val2, args.max_iter_sk, args.n_jobs, c_grid=c_grid)
        results['concat'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    if 'cuml' in args.method:
        t, bal, f1, mAP, per_class_ap = run_cuml(
            X_train, X_val, y_train, y_val, classes,
            args.n_trials, args.max_iter_cu,
            c_min=args.c_min, c_max=args.c_max, c_grid=c_grid)
        results['cuml'] = dict(time=t, bal=bal, f1=f1, mAP=mAP, per_class_ap=per_class_ap)

    print_summary(results)

    if args.out:
        save_results(results, args.out)


if __name__ == '__main__':
    main()
