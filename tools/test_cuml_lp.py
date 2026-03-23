#!/usr/bin/env python3
"""Smoke test: GPU-accelerated linear probe with cuML + optuna C search.

Tests that cuML LogisticRegression and optuna are working correctly by:
  1. Generating synthetic high-dim features (mimicking KRONOS output scale)
  2. Running Bayesian C search with optuna (5 trials, fast)
  3. Fitting the best model and reporting metrics

Usage:
    python tools/test_cuml_lp.py
"""

import numpy as np
import time

print("Importing cuml...", flush=True)
import cuml
from cuml.linear_model import LogisticRegression as cuLR
import cupy as cp
print(f"  cuml {cuml.__version__} OK", flush=True)

print("Importing optuna...", flush=True)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"  optuna {optuna.__version__} OK", flush=True)

from sklearn.metrics import balanced_accuracy_score, f1_score
# ── Synthetic data mimicking KRONOS18 output: 18×384 = 6912 dims ─────────────
N_TRAIN  = 50_000
N_VAL    = 10_000
N_DIMS   = 6_912   # 18 markers × 384
N_CLASSES = 16

rng = np.random.default_rng(42)
X_train = rng.standard_normal((N_TRAIN, N_DIMS)).astype(np.float32)
X_val   = rng.standard_normal((N_VAL,   N_DIMS)).astype(np.float32)
y_train = rng.integers(0, N_CLASSES, N_TRAIN).astype(np.int32)
y_val   = rng.integers(0, N_CLASSES, N_VAL).astype(np.int32)

print(f"\nData: train={N_TRAIN}×{N_DIMS}, val={N_VAL}×{N_DIMS}, {N_CLASSES} classes", flush=True)

# Move to GPU
X_train_gpu = cp.asarray(X_train)
X_val_gpu   = cp.asarray(X_val)
y_train_gpu = cp.asarray(y_train)

# ── Optuna C search (Bayesian, 5 trials) ─────────────────────────────────────
print("\nRunning optuna C search (5 trials)...", flush=True)
t0 = time.time()

def objective(trial):
    C = trial.suggest_float('C', 1e-6, 1e2, log=True)
    clf = cuLR(C=C, max_iter=1000, class_weight='balanced',
               output_type='numpy', verbose=False)
    clf.fit(X_train_gpu, y_train_gpu)
    preds = clf.predict(X_val_gpu)
    return float(f1_score(y_val, preds, average='macro', zero_division=0))

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=5, show_progress_bar=False)

best_C = study.best_params['C']
print(f"  Best C={best_C:.4g}  best_f1={study.best_value:.4f}  "
      f"({time.time()-t0:.1f}s)", flush=True)

# ── Final fit with best C ─────────────────────────────────────────────────────
print("\nFitting final model with best C...", flush=True)
t1 = time.time()
clf = cuLR(C=best_C, max_iter=1000, class_weight='balanced',
           output_type='numpy', verbose=False)
clf.fit(X_train_gpu, y_train_gpu)
preds = clf.predict(X_val_gpu)
bal_acc = balanced_accuracy_score(y_val, preds)
f1      = f1_score(y_val, preds, average='macro', zero_division=0)
print(f"  Balanced acc={bal_acc:.4f}  macro F1={f1:.4f}  ({time.time()-t1:.1f}s)", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Smoke test PASSED")
