#!/usr/bin/env python3
"""
Hyperparameter search for CIM + VICReg on CODEX_cHL.

Each trial writes a self-contained Python config, runs training in a
subprocess, and reads val balanced accuracy from metrics.json.

Usage
-----
    # run 30 trials (default)
    python hpo_CIM_VICReg_cHL.py

    # run 50 trials, show training output
    python hpo_CIM_VICReg_cHL.py --n-trials 50 --verbose

    # resume a previous study
    python hpo_CIM_VICReg_cHL.py --n-trials 20 --resume

    # print results without running new trials
    python hpo_CIM_VICReg_cHL.py --report

Dependencies
------------
    pip install optuna
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from pprint import pformat

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────────
MCA_ROOT  = Path(__file__).parent
MCA_PARENT = str(MCA_ROOT.parent)   # directory that contains the MCA package

# Load dataset paths directly from the dataset config (single source of truth)
_ns: dict = {}
exec((MCA_ROOT / "configs/_datasets_/CODEX_cHL.py").read_text(), _ns)

H5_FILEPATH       = _ns["h5_filepath"]
USED_MARKERS      = _ns["used_markers"]
TRAIN_IDX         = _ns["train_indicies"]
VAL_IDX           = _ns["val_indicies"]
IGNORE_ANNOTATION = _ns["ignore_annotation"]
N_MARKERS         = int(_ns["n_markers"])    # 41
CUTTER_SIZE       = int(_ns["cutter_size"])  # 24
PATCH_SIZE        = int(_ns["patch_size"])   # 32

ISILON     = "/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon"
WORK_BASE  = f"{ISILON}/src/MCA/z_RUNS/HPO_CIM_VICReg_cHL"
STUDY_DB   = f"sqlite:////home/simon_g/hpo_cHL.db"   # local disk, not NFS
STUDY_NAME = "CIM_VICReg_cHL"

# Fixed training budget per trial — keeps all trials comparable.
# Win: re-train with the full 10k schedule.
N_ITERS = 1000


# ── augmentation presets ───────────────────────────────────────────────────────

def _aug(strategy, cutter_size, strong):
    """Return a single-view augmentation pipeline list."""
    flip_prob = 0.5 if strong else 0.3
    steps = [dict(type="C_RandomFlip", prob=flip_prob, horizontal=True, vertical=True)]

    if strategy == "spatial":
        steps += [
            dict(type="C_RandomAffine",
                 angle=(0, 360),
                 scale=(0.66, 1.5) if strong else (0.9, 1.1),
                 shift=(-0.1, 0.1) if strong else (0, 0),
                 order=1),
            dict(type="C_RandomChannelShiftScale",
                 scale=(0.95, 1.05) if strong else (0.98, 1.02),
                 shift=(-0.01, 0.01) if strong else (-0.005, 0.005),
                 clip=True),
            dict(type="C_RandomNoise",
                 mean=(0, 0),
                 std=(0, 0.05) if strong else (0, 0.02),
                 clip=True),
        ]
    else:  # "high"
        steps += [
            dict(type="C_RandomAffine",
                 angle=(0, 360),
                 scale=(0.66, 1.5) if strong else (0.9, 1.1),
                 shift=(-0.1, 0.1) if strong else (0, 0),
                 order=1),
            dict(type="C_RandomChannelShiftScale",
                 scale=(0.33, 3.0) if strong else (0.9, 1.1),
                 shift=(-0.15, 0.15) if strong else (-0.05, 0.05),
                 clip=True),
            dict(type="C_RandomBackgroundGradient",
                 strength=(-0.15, 0.15) if strong else (0.0, 0.05),
                 clip=True),
            dict(type="C_RandomNoise",
                 mean=(0, 0),
                 std=(0, 0.05) if strong else (0, 0.02),
                 clip=True),
            dict(type="C_RandomChannelCopy",   copy_prob=0.05  if strong else 0.01),
            dict(type="C_RandomChannelMixup",  mixup_prob=0.05 if strong else 0.01),
            dict(type="C_RandomChannelDrop",   drop_prob=0.1   if strong else 0.025),
        ]

    steps += [
        dict(type="C_CentralCutter", size=cutter_size),
        dict(type="C_ToTensor"),
    ]
    return steps


# ── config builder ─────────────────────────────────────────────────────────────

def build_config(params, work_dir):
    """Return a complete mmengine config as a Python source string."""
    lr           = params["lr"]
    wd           = params["weight_decay"]
    batch_size   = params["batch_size"]
    stem_width   = params["stem_width"]
    sim_coeff    = params["sim_coeff"]
    std_coeff    = params["std_coeff"]
    aug_strategy = params["aug_strategy"]

    n_linear = 200
    n_cosine = N_ITERS - n_linear
    neck_in  = N_MARKERS * stem_width

    aug_strong = _aug(aug_strategy, CUTTER_SIZE, strong=True)
    aug_weak   = _aug(aug_strategy, CUTTER_SIZE, strong=False)
    val_aug    = [
        dict(type="C_CentralCutter", size=CUTTER_SIZE),
        dict(type="C_ToTensor"),
    ]

    train_pipeline = [
        dict(type="C_MultiView", n_views=[1, 1], transforms=[aug_strong, aug_weak]),
        dict(type="C_PackInputs"),
    ]
    val_pipeline = [
        dict(type="C_MultiView", n_views=[1], transforms=[val_aug]),
        dict(type="C_PackInputs"),
    ]

    dataset_kwargs = dict(
        h5_filepath       = H5_FILEPATH,
        used_markers      = USED_MARKERS,
        patch_size        = PATCH_SIZE,
        preprocess        = None,
        ignore_annotation = IGNORE_ANNOTATION,
    )

    train_dataset = dict(
        type       = "MCIDataset",
        mask_patch = True,
        pipeline   = train_pipeline,
        used_indicies = TRAIN_IDX,
        **dataset_kwargs,
    )

    cfg = dict(
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
        model=dict(
            type="MVVICReg",
            data_preprocessor=None,
            sim_coeff=sim_coeff,
            std_coeff=std_coeff,
            cov_coeff=1.0,
            gamma=1.0,
            backbone=dict(
                type="WideModel",
                in_channels=N_MARKERS,
                stem_width=stem_width,
                block_width=2,
                layer_config=[1, 1],
                late_fusion=False,
                drop_prob=0.05,
            ),
            neck=dict(
                type="NonLinearNeck",
                in_channels=neck_in,
                hid_channels=512,
                out_channels=512,
                num_layers=2,
                with_avg_pool=False,
            ),
        ),
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=lr, weight_decay=wd),
        ),
        param_scheduler=[
            dict(type="LinearLR",       start_factor=1e-4, by_epoch=False,
                 begin=0,        end=n_linear),
            dict(type="CosineAnnealingLR", T_max=n_cosine, eta_min=1e-6,
                 by_epoch=False, begin=n_linear, end=N_ITERS),
        ],
        train_cfg=dict(type="IterBasedTrainLoop", max_iters=N_ITERS),
        train_dataloader=dict(
            batch_size=batch_size,
            num_workers=8,
            sampler=dict(type="InfiniteSampler", shuffle=True),
            collate_fn=dict(type="default_collate"),
            drop_last=True,
            dataset=train_dataset,
        ),
        custom_hooks=[dict(
            type="EvaluateModel",
            train_indicies=TRAIN_IDX,
            val_indicies=VAL_IDX,
            pipeline=val_pipeline,
            dataset_kwargs=dataset_kwargs,
            short=True,
        )],
        default_hooks=dict(
            checkpoint=dict(type="CheckpointHook", by_epoch=False,
                            interval=N_ITERS, max_keep_ckpts=1),
            runtime_info=dict(type="RuntimeInfoHook"),
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", log_metric_by_epoch=False, interval=100),
            param_scheduler=dict(type="ParamSchedulerHook"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        ),
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

    # Serialise to Python source: one top-level assignment per key
    lines = []
    for key, val in cfg.items():
        lines.append(f"{key} = {pformat(val, width=120, sort_dicts=False)}")
    return "\n\n".join(lines) + "\n"


# ── Optuna objective ───────────────────────────────────────────────────────────

def objective(trial):
    params = dict(
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
        batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512]),
        stem_width   = trial.suggest_categorical("stem_width", [16, 32, 64]),
        sim_coeff    = trial.suggest_float("sim_coeff", 5.0, 50.0),
        std_coeff    = trial.suggest_float("std_coeff", 5.0, 50.0),
        aug_strategy = trial.suggest_categorical("aug_strategy", ["spatial", "high"]),
    )

    trial_dir = os.path.join(WORK_BASE, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save params for inspection
    with open(os.path.join(trial_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Write config to a temp file inside the trial dir
    cfg_src = build_config(params, trial_dir)
    cfg_path = os.path.join(trial_dir, "config.py")
    with open(cfg_path, "w") as f:
        f.write(cfg_src)

    # Run training as a subprocess
    runner_code = (
        f"import sys; sys.path.insert(0, {MCA_PARENT!r}); "
        f"from mmengine import Config; from mmengine.runner import Runner; "
        f"cfg = Config.fromfile({cfg_path!r}); Runner.from_cfg(cfg).train()"
    )

    t0 = time.monotonic()
    verbose = trial.study.user_attrs.get("verbose", False)
    gpu     = trial.study.user_attrs.get("gpu", "0")
    env     = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    try:
        if verbose:
            proc = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=env,
            )
            for line in proc.stdout:
                print(f"  [{trial.number}] {line}", end="", flush=True)
            proc.wait(timeout=7200)
            rc = proc.returncode
        else:
            result = subprocess.run(
                [sys.executable, "-c", runner_code],
                timeout=7200, capture_output=True, text=True,
                env=env,
            )
            rc = result.returncode
            if rc != 0:
                print(f"\n[trial {trial.number}] FAILED — last stderr lines:")
                print("\n".join(result.stderr.splitlines()[-20:]))

    except subprocess.TimeoutExpired:
        print(f"[trial {trial.number}] timed out — pruning")
        raise optuna.TrialPruned()

    elapsed = int(time.monotonic() - t0)

    if rc != 0:
        raise optuna.TrialPruned()

    metrics_path = os.path.join(trial_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[trial {trial.number}] metrics.json missing — pruning")
        raise optuna.TrialPruned()

    with open(metrics_path) as f:
        metrics = json.load(f)

    score = metrics["val"]["top1_balanced_accuracy"]
    print(
        f"[trial {trial.number:>3d}]  bal_acc={score:.4f}  "
        f"lr={params['lr']:.1e}  wd={params['weight_decay']:.1e}  "
        f"bs={params['batch_size']}  stem={params['stem_width']}  "
        f"sim={params['sim_coeff']:.1f}  std={params['std_coeff']:.1f}  "
        f"aug={params['aug_strategy']}  ({elapsed}s)"
    )
    return score


# ── reporting ──────────────────────────────────────────────────────────────────

def report(study):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("No completed trials yet.")
        return

    print(f"\n{'='*60}")
    print(f"Study : {study.study_name}")
    print(f"Trials: {len(done)} completed")
    print(f"Best  : val balanced accuracy = {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:<20} = {v}")

    print(f"\nTop-5 trials:")
    for rank, t in enumerate(sorted(done, key=lambda t: t.value, reverse=True)[:5], 1):
        print(f"  #{rank}  trial={t.number:04d}  val_bal_acc={t.value:.4f}")
        for k, v in t.params.items():
            print(f"       {k:<20} = {v}")

    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for k, v in imp.items():
            bar = "█" * max(1, int(v * 40))
            print(f"  {k:<20} {v:.3f}  {bar}")
    except Exception as e:
        print(f"(importances unavailable: {e})")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global WORK_BASE

    parser = argparse.ArgumentParser(description="HPO: CIM+VICReg on CODEX_cHL")
    parser.add_argument("--n-trials",   type=int,   default=30)
    parser.add_argument("--resume",     action="store_true",
                        help="Add trials to an existing study")
    parser.add_argument("--report",     action="store_true",
                        help="Print results and exit")
    parser.add_argument("--verbose",    action="store_true",
                        help="Stream subprocess training output")
    parser.add_argument("--study-name", type=str,   default=STUDY_NAME)
    parser.add_argument("--work-base",  type=str,   default=WORK_BASE)
    parser.add_argument("--db",         type=str,   default=STUDY_DB,
                        help="SQLAlchemy storage URL")
    parser.add_argument("--gpu",        type=str,   default="0",
                        help="CUDA device id(s) passed to CUDA_VISIBLE_DEVICES (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.work_base, exist_ok=True)
    WORK_BASE = args.work_base

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )
    study.set_user_attr("verbose", args.verbose)
    study.set_user_attr("gpu", args.gpu)

    if args.report:
        report(study)
        return

    existing = len(study.trials)
    if existing > 0 and not args.resume:
        print(
            f"Study '{args.study_name}' already has {existing} trials. "
            "Pass --resume to add more, or use a different --study-name."
        )
        return

    print(f"Starting HPO: {args.n_trials} trials × {N_ITERS} iters each")
    print(f"Results → {args.work_base}\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print()
    report(study)


if __name__ == "__main__":
    main()
