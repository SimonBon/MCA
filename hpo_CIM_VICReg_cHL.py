#!/usr/bin/env python3
"""Hyperparameter Optimization for CIM + VICReg on CODEX_cHL.

Uses Optuna with TPE sampler and SQLite storage for persistence across sessions.
Each trial builds a complete mmengine config, runs training in an isolated
subprocess, and reads the linear-probe balanced accuracy from metrics.json.

Training budget per trial
─────────────────────────
HPO does NOT run the full 10 000-iter training (train_cfg_long.py) – that would
take too long per trial.  Instead every trial uses a FIXED budget of 1 000 iters
(matching the standard short config: 200 warmup + 800 cosine).  This keeps all
trials comparable.  Only after HPO is done do you re-train the winner with the
long schedule.  Override with --n-iters if you want a different budget.

Usage:
    # 50 trials at 1 000 iters each (default)
    python hpo_CIM_VICReg_cHL.py --n-trials 50

    # Fast sweep at 500 iters (rougher signal, ~2× faster)
    python hpo_CIM_VICReg_cHL.py --n-trials 100 --n-iters 500

    # Resume an existing study
    python hpo_CIM_VICReg_cHL.py --n-trials 20 --resume

    # Stream training output + live leaderboard
    python hpo_CIM_VICReg_cHL.py --n-trials 50 --verbose

    # Print best results and parameter importances
    python hpo_CIM_VICReg_cHL.py --report

Dependencies:
    pip install optuna optuna-dashboard
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from pprint import pformat
import time
import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Set to True via --verbose flag; read in objective() and verbose_callback()
_VERBOSE = False

# ============================================================
# PATHS – loaded directly from the cHL dataset config so there
# is a single source of truth and no path duplication.
# ============================================================
_MCA_ROOT   = Path(__file__).parent          # .../MCA/
_cHL_ns: dict = {}
exec((_MCA_ROOT / "configs/_datasets_/CODEX_cHL.py").read_text(), _cHL_ns)

H5_FILEPATH       = _cHL_ns["h5_filepath"]
USED_MARKERS      = _cHL_ns["used_markers"]
TRAIN_INDICIES    = _cHL_ns["train_indicies"]
VAL_INDICIES      = _cHL_ns["val_indicies"]
IGNORE_ANNOTATION = _cHL_ns["ignore_annotation"]
N_MARKERS         = _cHL_ns["n_markers"]       # 41
CUTTER_SIZE       = _cHL_ns["cutter_size"]     # 24
PATCH_SIZE        = _cHL_ns["patch_size"]      # 32

# HPO output dir (trial checkpoints, metrics.json, configs)
_ISILON       = "/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon"
HPO_WORK_BASE = f"{_ISILON}/src/MCA/z_RUNS/HPO_CIM_VICReg_cHL"

# SQLite DB stored next to this script for easy access
STUDY_DB   = f"sqlite:///{_MCA_ROOT / 'hpo_CIM_VICReg_cHL.db'}"
STUDY_NAME = "CIM_VICReg_cHL_v1"

# Fixed training budget per HPO trial.
# 1 000 iters  = standard short config (train_cfg.py: 200 warmup + 800 cosine).
# All trials use the same budget so scores are directly comparable.
# The winning config is then re-trained with train_cfg_long.py (10 000 iters).
HPO_N_ITERS = 1000

# Mutable at runtime by --n-iters; read by sample_params() and objective()
_HPO_N_ITERS_CURRENT = HPO_N_ITERS

# ============================================================
# AUGMENTATION PIPELINE BUILDERS
# ============================================================

def _aug_spatial(cutter_size: int, strong: bool) -> list:
    """Spatial-only: flip + affine + mild channel normalisation + noise.
    No channel drop / copy / mixup → best strategy per PROGRESS.md (+1.1pp).
    """
    if strong:
        return [
            dict(type="C_RandomFlip", prob=0.5, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 360), scale=(0.66, 1.5),
                 shift=(-0.1, 0.1), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.95, 1.05),
                 shift=(-0.01, 0.01), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(-0.15, 0.15), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.05), clip=True),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]
    else:
        return [
            dict(type="C_RandomFlip", prob=0.3, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 360), scale=(0.9, 1.1),
                 shift=(0, 0), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.98, 1.02),
                 shift=(-0.005, 0.005), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(0.0, 0.05), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.02), clip=True),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]


def _aug_low(cutter_size: int, strong: bool) -> list:
    """Low channel augmentation: conservative drop / copy / mixup probabilities."""
    if strong:
        return [
            dict(type="C_RandomFlip", prob=0.5, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 90), scale=(0.9, 1.1),
                 shift=(0, 0), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.8, 1.2),
                 shift=(-0.05, 0.05), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(-0.05, 0.05), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.02), clip=True),
            dict(type="C_RandomChannelCopy", copy_prob=0.02),
            dict(type="C_RandomChannelMixup", mixup_prob=0.02),
            dict(type="C_RandomChannelDrop", drop_prob=0.05),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]
    else:
        return [
            dict(type="C_RandomFlip", prob=0.3, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 45), scale=(0.95, 1.05),
                 shift=(0, 0), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.9, 1.1),
                 shift=(-0.02, 0.02), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(0.0, 0.02), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.01), clip=True),
            dict(type="C_RandomChannelCopy", copy_prob=0.005),
            dict(type="C_RandomChannelMixup", mixup_prob=0.005),
            dict(type="C_RandomChannelDrop", drop_prob=0.01),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]


def _aug_high(cutter_size: int, strong: bool) -> list:
    """High channel augmentation: current default for CIM_VICReg_cHL."""
    if strong:
        return [
            dict(type="C_RandomFlip", prob=0.5, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 360), scale=(0.66, 1.5),
                 shift=(-0.1, 0.1), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.33, 3),
                 shift=(-0.15, 0.15), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(-0.15, 0.15), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.05), clip=True),
            dict(type="C_RandomChannelCopy", copy_prob=0.05),
            dict(type="C_RandomChannelMixup", mixup_prob=0.05),
            dict(type="C_RandomChannelDrop", drop_prob=0.1),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]
    else:
        return [
            dict(type="C_RandomFlip", prob=0.3, horizontal=True, vertical=True),
            dict(type="C_RandomAffine", angle=(0, 360), scale=(0.9, 1.1),
                 shift=(0, 0), order=1),
            dict(type="C_RandomChannelShiftScale", scale=(0.9, 1.1),
                 shift=(-0.05, 0.05), clip=True),
            dict(type="C_RandomBackgroundGradient", strength=(0.0, 0.05), clip=True),
            dict(type="C_RandomNoise", mean=(0, 0), std=(0, 0.02), clip=True),
            dict(type="C_RandomChannelCopy", copy_prob=0.01),
            dict(type="C_RandomChannelMixup", mixup_prob=0.01),
            dict(type="C_RandomChannelDrop", drop_prob=0.025),
            dict(type="C_CentralCutter", size=cutter_size),
            dict(type="C_ToTensor"),
        ]


_AUG_BUILDERS = {
    "spatial": _aug_spatial,
    "low":     _aug_low,
    "high":    _aug_high,
}

# ============================================================
# CONFIG BUILDER
# ============================================================

def build_cfg_dict(params: dict, work_dir: str) -> dict:
    """Build a complete mmengine config dict from the sampled HPO params.

    Parameters
    ----------
    params : dict
        Keys (see ``sample_params`` below for full list).
    work_dir : str
        Per-trial output directory (checkpoints, metrics.json, etc.).
    """
    lr            = params["lr"]
    weight_decay  = params["weight_decay"]
    n_linear      = params["n_linear"]
    n_cosine      = params["n_cosine"]   # = HPO_N_ITERS - n_linear (fixed budget)
    batch_size    = params["batch_size"]
    stem_width    = params["stem_width"]
    drop_prob     = params["drop_prob"]
    layer_config  = params["layer_config"]
    late_fusion   = params["late_fusion"]
    sim_coeff     = params["sim_coeff"]
    std_coeff     = params["std_coeff"]
    cov_coeff     = params["cov_coeff"]
    proj_dim      = params["proj_dim"]
    aug_strategy  = params["aug_strategy"]

    n_iters = n_linear + n_cosine        # always == HPO_N_ITERS (or --n-iters)
    neck_in = N_MARKERS * stem_width     # e.g. 41 * 32 = 1312

    aug_fn = _AUG_BUILDERS[aug_strategy]
    train_aug_strong = aug_fn(CUTTER_SIZE, strong=True)
    train_aug_weak   = aug_fn(CUTTER_SIZE, strong=False)

    val_augmentation = [
        dict(type="C_CentralCutter", size=CUTTER_SIZE),
        dict(type="C_ToTensor"),
    ]
    val_pipeline = [
        dict(type="C_MultiView", n_views=[1], transforms=[val_augmentation]),
        dict(type="C_PackInputs"),
    ]
    train_pipeline = [
        dict(type="C_MultiView", n_views=[1, 1],
             transforms=[train_aug_strong, train_aug_weak]),
        dict(type="C_PackInputs"),
    ]

    # Paths come from configs/_datasets_/CODEX_cHL.py (single source of truth)
    dataset_kwargs = dict(
        h5_filepath=H5_FILEPATH,
        used_markers=USED_MARKERS,
        patch_size=PATCH_SIZE,
        preprocess=None,
        ignore_annotation=IGNORE_ANNOTATION,
        used_indicies=TRAIN_INDICIES,  # overridden per-split by EvaluateModel hook
    )

    train_dataset = dict(
        type="MCIDataset",
        mask_patch=True,
        pipeline=train_pipeline,
        **dataset_kwargs,
    )

    cfg = dict(
        # ---- imports ----
        custom_imports=dict(
            imports=[
                "MCA.configs._datasets_",
                "MCA.src.dataset",
                "MCA.src.transforms",
                "MCA.src.SimCLR",
                "MCA.src.VICReg",
                "MCA.src.BYOL",
                "MCA.src.models",
                "MCA.src.models_attention",
                "MCA.src.models_early_fusion",
                "MCA.src.MCM",
                "MCA.src.val_hook",
            ],
            allow_failed_imports=False,
        ),
        default_scope="mmselfsup",

        # ---- model ----
        model=dict(
            type="MVVICReg",
            data_preprocessor=None,
            sim_coeff=sim_coeff,
            std_coeff=std_coeff,
            cov_coeff=cov_coeff,
            gamma=1.0,
            backbone=dict(
                type="WideModel",
                in_channels=N_MARKERS,
                stem_width=stem_width,
                block_width=2,
                layer_config=layer_config,
                late_fusion=late_fusion,
                drop_prob=drop_prob,
            ),
            neck=dict(
                type="NonLinearNeck",
                in_channels=neck_in,
                hid_channels=proj_dim,
                out_channels=proj_dim,
                num_layers=2,
                with_avg_pool=False,
            ),
        ),

        # ---- optimizer ----
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=lr, weight_decay=weight_decay),
        ),

        # ---- lr schedule ----
        param_scheduler=[
            dict(type="LinearLR", start_factor=1e-4,
                 by_epoch=False, begin=0, end=n_linear),
            dict(type="CosineAnnealingLR", T_max=n_cosine, eta_min=1e-6,
                 by_epoch=False, begin=n_linear, end=n_iters),
        ],

        # ---- training loop ----
        train_cfg=dict(type="IterBasedTrainLoop", max_iters=n_iters),
        train_dataloader=dict(
            batch_size=batch_size,
            num_workers=8,
            sampler=dict(type="InfiniteSampler", shuffle=True),
            collate_fn=dict(type="default_collate"),
            drop_last=True,
            dataset=train_dataset,
        ),

        # ---- linear probe validation ----
        # short=True caps embedding extraction at 25k cells for speed during HPO
        custom_hooks=[dict(
            type="EvaluateModel",
            train_indicies=TRAIN_INDICIES,
            val_indicies=VAL_INDICIES,
            short=True,
            pipeline=val_pipeline,
            dataset_kwargs=dataset_kwargs,
        )],

        # ---- hooks / logging ----
        default_hooks=dict(
            # Save only final checkpoint to save disk space during HPO
            checkpoint=dict(type="CheckpointHook", by_epoch=False,
                            interval=n_iters, max_keep_ckpts=1),
            runtime_info=dict(type="RuntimeInfoHook"),
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", log_metric_by_epoch=False, interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        ),

        # ---- env ----
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
            dist_cfg=dict(backend="nccl"),
        ),
        log_processor=dict(
            window_size=1,
            custom_cfg=[dict(data_src="", method="mean", window_size="global")],
        ),
        vis_backends=[dict(type="LocalVisBackend")],
        visualizer=dict(
            type="SelfSupVisualizer",
            vis_backends=[dict(type="LocalVisBackend")],
            name="visualizer",
        ),
        log_level="WARNING",
        load_from=None,
        resume=False,
        work_dir=work_dir,
        launcher="none",
    )
    return cfg


# ============================================================
# CONFIG → PYTHON FILE  (for mmengine Config.fromfile)
# ============================================================

def cfg_dict_to_py(cfg: dict) -> str:
    """Serialise a config dict to a valid Python source file string.
    Uses pformat so all values are pure Python literals (no imports needed).
    """
    lines = []
    for key, value in cfg.items():
        lines.append(f"{key} = {pformat(value, width=120, sort_dicts=False)}")
    return "\n\n".join(lines) + "\n"


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

def sample_params(trial: optuna.Trial) -> dict:
    """Define and sample the full search space for one Optuna trial.

    Rationale for each group:
    ─────────────────────────
    Optimiser:
      lr           log-uniform [5e-5, 5e-3]  – current best 3e-4 is near centre
      weight_decay log-uniform [1e-4, 0.2]   – AdamW regularisation strength

    Schedule:
      n_linear   warmup fraction of the fixed HPO_N_ITERS budget.
                 n_cosine is derived as HPO_N_ITERS - n_linear so all trials
                 train for exactly the same number of iterations (fair comparison).
                 DO NOT put n_cosine in the search space.

    Dataloader:
      batch_size  – larger batches improve VICReg covariance estimation
                    but increase memory (must fit VRAM)

    CIM backbone:
      stem_width   – features per marker (main capacity knob for CIM)
      drop_prob    – Dropout2d for regularisation
      layer_config – network depth / stages
      late_fusion  – allow cross-channel interaction after spatial processing

    VICReg loss:
      sim_coeff  – invariance term weight (currently 25)
      std_coeff  – variance term weight  (currently 25)
      cov_coeff  – covariance term weight (currently 1, least critical)
        NOTE: sim/std ratio matters more than absolute scale

    Projector:
      proj_dim  – hidden & output dim of the NonLinearNeck MLP

    Augmentation:
      aug_strategy – PROGRESS.md shows spatial >> high (+1.1pp bal acc)
                     so spatial is recommended but we confirm it here
    """
    n_linear = trial.suggest_categorical("n_linear", [100, 200, 500])
    # Budget is fixed; n_cosine fills the remainder so all trials are comparable
    n_cosine = _HPO_N_ITERS_CURRENT - n_linear

    return dict(
        # ── optimiser ──────────────────────────────────────────────────────
        lr           = trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.2, log=True),

        # ── schedule ───────────────────────────────────────────────────────
        n_linear = n_linear,
        n_cosine = n_cosine,   # derived; not a search param

        # ── dataloader ─────────────────────────────────────────────────────
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512]),

        # ── CIM backbone ───────────────────────────────────────────────────
        stem_width   = trial.suggest_categorical("stem_width", [16, 32, 64]),
        drop_prob    = trial.suggest_float("drop_prob", 0.0, 0.2),
        layer_config = json.loads(trial.suggest_categorical(
            "layer_config", ["[1]", "[1,1]", "[2,2]"]
        )),
        late_fusion  = trial.suggest_categorical("late_fusion", [False, True]),

        # ── VICReg loss ────────────────────────────────────────────────────
        sim_coeff = trial.suggest_float("sim_coeff", 5.0, 50.0),
        std_coeff = trial.suggest_float("std_coeff", 5.0, 50.0),
        cov_coeff = trial.suggest_float("cov_coeff", 0.1, 5.0, log=True),

        # ── projector ──────────────────────────────────────────────────────
        proj_dim = trial.suggest_categorical("proj_dim", [256, 512, 1024]),

        # ── augmentation ───────────────────────────────────────────────────
        aug_strategy = trial.suggest_categorical(
            "aug_strategy", ["spatial", "low", "high"]
        ),
    )


# ============================================================
# OBJECTIVE
# ============================================================

# Inline runner script executed by each subprocess trial.
_RUNNER_SCRIPT = """\
import sys
sys.path.insert(0, "{mca_root}")
from mmengine import Config
from mmengine.runner import Runner
cfg = Config.fromfile("{config_path}")
runner = Runner.from_cfg(cfg)
runner.train()
"""


def objective(trial: optuna.Trial) -> float:
    """Run one CIM+VICReg training trial and return val balanced accuracy.

    Training budget is fixed to _HPO_N_ITERS_CURRENT (default 1 000 iters)
    so all trials are directly comparable.  Only the warmup fraction (n_linear)
    is searched; n_cosine fills the remainder automatically.
    """
    params = sample_params(trial)

    trial_work_dir = os.path.join(HPO_WORK_BASE, f"trial_{trial.number:04d}")
    os.makedirs(trial_work_dir, exist_ok=True)

    cfg_dict = build_cfg_dict(params, trial_work_dir)
    cfg_py   = cfg_dict_to_py(cfg_dict)

    # Write temp config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=trial_work_dir,
        prefix="hpo_cfg_"
    ) as f:
        f.write(cfg_py)
        config_path = f.name

    # Save params for easy inspection
    with open(os.path.join(trial_work_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    mca_root = str(Path(__file__).parent.parent)  # directory containing MCA package
    runner_code = _RUNNER_SCRIPT.format(
        mca_root=mca_root,
        config_path=config_path,
    )

    t0 = time.monotonic()
    cmd = [sys.executable, "-c", runner_code]

    try:
        if _VERBOSE:
            # Stream subprocess output line-by-line so you can follow training
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stdout_lines = []
            for line in proc.stdout:
                print(f"  | {line}", end="", flush=True)
                stdout_lines.append(line)
            proc.wait(timeout=7200)
            returncode = proc.returncode
            stderr_tail = ""  # merged into stdout above
        else:
            result = subprocess.run(
                cmd, timeout=7200, capture_output=True, text=True
            )
            returncode   = result.returncode
            stderr_tail  = result.stderr[-3000:]
            stdout_lines = result.stdout.splitlines(keepends=True)

        elapsed = datetime.timedelta(seconds=int(time.monotonic() - t0))

        if returncode != 0:
            print(f"\n[Trial {trial.number}] FAILED (returncode={returncode}, elapsed={elapsed})")
            if stderr_tail:
                print("STDERR:", stderr_tail)
            raise optuna.exceptions.TrialPruned()

        metrics_path = os.path.join(trial_work_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"\n[Trial {trial.number}] metrics.json not found – pruning.")
            raise optuna.exceptions.TrialPruned()

        with open(metrics_path) as f:
            metrics = json.load(f)

        val_bal_acc = metrics["val"]["top1_balanced_accuracy"]
        print(
            f"\n[Trial {trial.number}] DONE  val_bal_acc={val_bal_acc:.4f}  elapsed={elapsed}\n"
            f"  aug={params['aug_strategy']}  lr={params['lr']:.2e}  "
            f"wd={params['weight_decay']:.2e}  stem={params['stem_width']}  "
            f"sim={params['sim_coeff']:.1f}  std={params['std_coeff']:.1f}  "
            f"cov={params['cov_coeff']:.2f}  proj={params['proj_dim']}  "
            f"bs={params['batch_size']}  drop={params['drop_prob']:.2f}  "
            f"layers={params['layer_config']}  fusion={params['late_fusion']}"
        )
        return val_bal_acc

    except subprocess.TimeoutExpired:
        elapsed = datetime.timedelta(seconds=int(time.monotonic() - t0))
        print(f"\n[Trial {trial.number}] TIMED OUT after {elapsed} – pruning.")
        raise optuna.exceptions.TrialPruned()


# ============================================================
# VERBOSE CALLBACK  (fires after every trial)
# ============================================================

def verbose_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print a live leaderboard after each completed trial."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    n_total   = len(study.trials)

    sep = "─" * 68
    print(f"\n{sep}")
    print(
        f"  After trial {trial.number:>4d} / running total: "
        f"{len(completed)} complete, {len(pruned)} pruned, {n_total} total"
    )

    if trial.state == optuna.trial.TrialState.COMPLETE:
        is_best = trial.value == study.best_value
        flag = "  ★ NEW BEST" if is_best else ""
        print(f"  This trial : val_bal_acc = {trial.value:.4f}{flag}")

    if completed:
        print(f"  Study best : val_bal_acc = {study.best_value:.4f}  "
              f"(trial {study.best_trial.number:04d})")

        # Mini leaderboard: top-5
        top = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        print(f"\n  {'Rank':>4}  {'Trial':>5}  {'val_bal_acc':>11}  Key params")
        for rank, t in enumerate(top, 1):
            p = t.params
            print(
                f"  {rank:>4}  {t.number:>5}  {t.value:>11.4f}  "
                f"aug={p.get('aug_strategy','?'):<8} "
                f"lr={p.get('lr', 0):.2e}  "
                f"stem={p.get('stem_width','?')}  "
                f"sim={p.get('sim_coeff', 0):.1f}  "
                f"std={p.get('std_coeff', 0):.1f}"
            )

    print(sep)


# ============================================================
# REPORTING
# ============================================================

def report(study: optuna.Study) -> None:
    """Print a human-readable summary of the study results."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        print("No completed trials yet.")
        return

    print(f"\n{'='*60}")
    print(f"Study: {study.study_name}")
    print(f"Completed trials: {len(completed)}")
    print(f"Best val balanced accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} = {v}")

    print(f"\nTop-5 trials:")
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    for rank, t in enumerate(top5, 1):
        print(f"  #{rank}  trial={t.number:04d}  val_bal_acc={t.value:.4f}")
        for k, v in t.params.items():
            print(f"         {k:20s} = {v}")
        print()

    # Parameter importances (requires sklearn)
    try:
        importances = optuna.importance.get_param_importances(study)
        print("Parameter importances (FAnova):")
        for k, v in importances.items():
            bar = "█" * int(v * 40)
            print(f"  {k:20s} {v:.3f}  {bar}")
    except Exception as e:
        print(f"(Could not compute importances: {e})")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global HPO_WORK_BASE, _VERBOSE, _HPO_N_ITERS_CURRENT

    parser = argparse.ArgumentParser(
        description="HPO for CIM+VICReg on CODEX_cHL"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials to run (default: 50)",
    )
    parser.add_argument(
        "--n-iters", type=int, default=None,
        help=f"Override total training iters per trial (default: {HPO_N_ITERS}). "
             "All trials use this fixed budget so they are comparable. "
             "Use a smaller value (e.g. 500) for faster but noisier sweeps.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Stop study after this many seconds (wall clock). "
             "Useful on clusters with job time limits.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel Optuna workers (each runs its own subprocess). "
             "Share the same SQLite study. Requires --n-jobs > 1 separately "
             "launched processes or set n_jobs here for thread-level parallelism.",
    )
    parser.add_argument(
        "--study-name", type=str, default=STUDY_NAME,
        help=f"Optuna study name (default: {STUDY_NAME})",
    )
    parser.add_argument(
        "--storage", type=str, default=STUDY_DB,
        help="SQLAlchemy storage URL (default: local SQLite DB)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing study from storage and add more trials.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print study results and exit (no new trials).",
    )
    parser.add_argument(
        "--work-base", type=str, default=HPO_WORK_BASE,
        help="Root directory for trial outputs.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help=(
            "Stream subprocess training output in real-time and print a "
            "live leaderboard after every trial."
        ),
    )
    args = parser.parse_args()

    # Allow CLI overrides of module-level flags / paths
    HPO_WORK_BASE         = args.work_base
    _VERBOSE              = args.verbose
    _HPO_N_ITERS_CURRENT  = args.n_iters if args.n_iters else HPO_N_ITERS

    if _VERBOSE:
        # Let mmengine print normally; suppress only optuna's own INFO spam
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    os.makedirs(HPO_WORK_BASE, exist_ok=True)

    sampler = TPESampler(seed=42, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,   # always safe: creates if absent, loads if present
        sampler=sampler,
        pruner=pruner,
    )

    if args.report:
        report(study)
        return

    if not args.resume and len(study.trials) > 0:
        print(
            f"[WARNING] Study '{args.study_name}' already has "
            f"{len(study.trials)} trials. "
            "Pass --resume to continue adding trials, or use a different "
            "--study-name to start fresh."
        )

    callbacks = [verbose_callback]  # always register; prints leaderboard each trial

    print(
        f"Training budget per trial: {_HPO_N_ITERS_CURRENT} iters  "
        f"(use --n-iters to change)"
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
        show_progress_bar=not _VERBOSE,  # tqdm bar is redundant when verbose
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("HPO complete.")
    report(study)


if __name__ == "__main__":
    main()
