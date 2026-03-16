#!/usr/bin/env python3
"""Generate all paper experiment configs.

Creates:
  configs/_experiments_/paper/<DATASET>/<MODEL>_VICReg[_fold<K>].py

Datasets:
  CODEX_cHL       — single split (no CV), annotation_map for class merges
  MIBI_TNBC       — 5-fold patient CV (cv_splits_paper/split_k/)
  IMC_NB_TumorSub — 5-fold patient CV (cv_splits/split_k/)

Models per dataset:
  CIM, CIM_LateFusion, CIM_Funnel_Large, ResNet
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
NB   = '/nobackup/lab_taschner-mandl/simongutwein'
RUNS = f'{NB}/z_RUNS/paper_clean'

DATASETS = {
    'CODEX_cHL': {
        'cutter_size': 24,
        'data': f'{NB}/h5_files/CODEX_cHL',
        'h5':   'CODEX_cHL.h5',
        'style': 'kwargs',   # uses dataset_kwargs pattern
        'splits': None,      # single split — train.txt / test.txt
        'annotation_map': "{'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'}",
    },
    'CODEX_cHL_KRONOS18': {
        'cutter_size': 24,
        'data': f'{NB}/h5_files/CODEX_cHL',
        'h5':   'CODEX_cHL.h5',
        'markers_file': 'used_markers_KRONOS18.txt',
        'style': 'kwargs',
        'splits': None,
        'annotation_map': "{'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'}",
    },
    'MIBI_TNBC': {
        'cutter_size': 20,
        'data': f'{NB}/h5_files/MIBI_TNBC',
        'h5':   'MIBI_TNBC.h5',
        'style': 'direct',
        'splits': 'cv_splits_paper',
        'annotation_map': None,
    },
    'IMC_NB_TumorSub': {
        'cutter_size': 12,
        'data': f'{NB}/h5_files/IMC_NB_TumorSub',
        'h5':   'IMC_NB_TumorSub.h5',
        'style': 'direct',
        'splits': 'cv_splits',
        'annotation_map': None,
    },
}

MODELS = {
    'CIM':              {'backbone': 'CIM.py',               'lars': False, 'neck': 'markers_x_feat'},
    'CIM_LateFusion':   {'backbone': 'CIM_LateFusion_MLP.py', 'lars': False, 'neck': 'markers_x_feat'},
    'CIM_Funnel_Large': {'backbone': 'CIM_Funnel_Large.py',   'lars': True,  'neck': 'mix_channels'},
    'ResNet':           {'backbone': 'ResNet.py',             'lars': False, 'neck': '256'},
}

N_FOLDS = 5
OUT_ROOT = Path(__file__).parent.parent / 'configs' / '_experiments_' / 'paper'


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_paths(ds, fold):
    """Return (train_txt, test_txt) for a given fold (None = single split)."""
    data = ds['data']
    if fold is None:
        return f'{data}/train.txt', f'{data}/test.txt'
    return (
        f'{data}/{ds["splits"]}/split_{fold}/train.txt',
        f'{data}/{ds["splits"]}/split_{fold}/test.txt',
    )


def _path_override_block(ds_name, ds, train_txt, test_txt):
    """Lines that override _base_ paths to nobackup, style-aware."""
    data, h5 = ds['data'], ds['h5']
    markers_file = ds.get('markers_file', 'used_markers.txt')
    lines = [
        '# ── Override paths to nobackup ──────────────────────────────────────────────',
        f"_base_.h5_filepath    = '{data}/{h5}'",
        f"_base_.used_markers   = '{data}/{markers_file}'",
        f"_base_.train_indicies = '{train_txt}'",
        f"_base_.test_indicies  = '{test_txt}'",
    ]
    if ds['style'] == 'kwargs':
        lines += [
            "_base_.dataset_kwargs['h5_filepath']  = _base_.h5_filepath",
            "_base_.dataset_kwargs['used_markers'] = _base_.used_markers",
        ]
    else:
        lines += [
            "_base_.dataset['h5_filepath']  = _base_.h5_filepath",
            "_base_.dataset['used_markers'] = _base_.used_markers",
        ]
    return '\n'.join(lines)


def _train_dataset_block(ds):
    """Build train_dataset from base dataset dict (style-aware)."""
    if ds['style'] == 'kwargs':
        return (
            "train_dataset = deepcopy(_base_.dataset)\n"
            "train_dataset.update(_base_.dataset_kwargs)\n"
            "train_dataset['used_indicies'] = _base_.train_indicies\n"
            "train_dataset['pipeline']      = _base_.train_pipeline\n"
            "train_dataset['mask_patch']    = mask_patch"
        )
    else:
        return (
            "train_dataset = deepcopy(_base_.dataset)\n"
            "train_dataset['used_indicies'] = _base_.train_indicies\n"
            "train_dataset['pipeline']      = _base_.train_pipeline\n"
            "train_dataset['mask_patch']    = mask_patch"
        )


def _annotation_map_line(ds):
    if ds['annotation_map']:
        return f"_base_.custom_hooks[0].annotation_map = {ds['annotation_map']}\n"
    return ''


# ── Standard config (CIM / CIM_LateFusion / ResNet — AdamW 1k iters) ──────────

def make_standard_config(ds_name, ds, model_name, model, fold):
    train_txt, test_txt = _split_paths(ds, fold)
    fold_str = f'/fold_{fold}' if fold is not None else ''
    work_dir = f'{RUNS}/{ds_name}/{model_name}{fold_str}'

    path_block    = _path_override_block(ds_name, ds, train_txt, test_txt)
    dataset_block = _train_dataset_block(ds)
    amap_line     = _annotation_map_line(ds)

    if model['neck'] == 'markers_x_feat':
        neck_line = '_base_.model.neck.in_channels     = _base_.n_markers * _base_.features_per_marker'
    else:
        neck_line = f'_base_.model.neck.in_channels     = {model["neck"]}'

    return f"""\
from copy import deepcopy

_base_ = [
    '../../../_base_/default.py',
    '../../../_augmentations_/high.py',
    '../../../_base_/train_cfg.py',
    '../../../_base_/val_cfg.py',
    '../../../_datasets_/{ds_name}.py',
    '../../../_backbones_/{model["backbone"]}',
    '../../../_algorithms_/VICReg.py',
]

batch_size  = 256
num_workers = 16
mask_patch  = True

{path_block}

_base_.val_augmentation[0].size = _base_.cutter_size
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]

_base_.train_aug_strong[-2].size = _base_.cutter_size
_base_.train_aug_weak[-2].size   = _base_.cutter_size
_base_.train_pipeline[0].transforms = [_base_.train_aug_strong, _base_.train_aug_weak]

{dataset_block}

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=train_dataset,
)

dataset_kwargs = dict(
    h5_filepath=_base_.h5_filepath,
    used_markers=_base_.used_markers,
    patch_size=_base_.patch_size,
    ignore_annotation=_base_.ignore_annotation,
)

_base_.custom_hooks[0].type           = 'EvaluateModelRich'
_base_.custom_hooks[0].n_jobs         = 8
_base_.custom_hooks[0].train_indicies = _base_.train_indicies
_base_.custom_hooks[0].val_indicies   = _base_.test_indicies
_base_.custom_hooks[0].pipeline       = _base_.val_pipeline
_base_.custom_hooks[0].dataset_kwargs = dataset_kwargs
{amap_line}
_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
{neck_line}

work_dir = '{work_dir}'
"""


# ── Funnel config (CIM_Funnel_Large — LARS 16k iters, inline augs) ─────────────

def make_funnel_config(ds_name, ds, fold):
    train_txt, test_txt = _split_paths(ds, fold)
    fold_str = f'/fold_{fold}' if fold is not None else ''
    work_dir = f'{RUNS}/{ds_name}/CIM_Funnel_Large{fold_str}'

    cs      = ds['cutter_size']
    cs_weak = cs - 2

    path_block    = _path_override_block(ds_name, ds, train_txt, test_txt)
    amap_line     = _annotation_map_line(ds)

    if ds['style'] == 'kwargs':
        dataset_block = (
            "train_dataset = deepcopy(_base_.dataset)\n"
            "train_dataset.update(_base_.dataset_kwargs)\n"
            "train_dataset['used_indicies'] = _base_.train_indicies\n"
            "train_dataset['pipeline']      = train_pipeline\n"
            "train_dataset['mask_patch']    = mask_patch"
        )
    else:
        dataset_block = (
            "train_dataset = deepcopy(_base_.dataset)\n"
            "train_dataset['used_indicies'] = _base_.train_indicies\n"
            "train_dataset['pipeline']      = train_pipeline\n"
            "train_dataset['mask_patch']    = mask_patch"
        )

    return f"""\
from copy import deepcopy

_base_ = [
    '../../../_base_/default.py',
    '../../../_base_/val_cfg.py',
    '../../../_datasets_/{ds_name}.py',
    '../../../_backbones_/CIM_Funnel_Large.py',
    '../../../_algorithms_/VICReg.py',
]

batch_size  = 128
num_workers = 16
mask_patch  = True

{path_block}

# ── Optimizer: LARS, 16k iters ────────────────────────────────────────────────
n_linear      = 800
n_cosine      = 15200
optimizer     = dict(type='LARS', lr=0.3, momentum=0.9, weight_decay=1e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_cfg     = dict(type='IterBasedTrainLoop', max_iters=n_linear + n_cosine)
param_scheduler = [
    dict(type='LinearLR',          start_factor=1e-4, by_epoch=False, begin=0,        end=n_linear),
    dict(type='CosineAnnealingLR', T_max=n_cosine,    by_epoch=False, begin=n_linear, end=n_linear + n_cosine, eta_min=0.03),
]

# ── Augmentations ─────────────────────────────────────────────────────────────
train_aug_strong = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size={cs}),
    dict(type='C_ToTensor'),
]

train_aug_weak = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size={cs_weak}),
    dict(type='C_ToTensor'),
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[train_aug_strong, train_aug_weak]),
    dict(type='C_PackInputs'),
]

_base_.val_augmentation[0].size = {cs_weak}
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]

# ── Dataset ───────────────────────────────────────────────────────────────────
{dataset_block}

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=train_dataset,
)

# ── Eval hook ─────────────────────────────────────────────────────────────────
dataset_kwargs = dict(
    h5_filepath=_base_.h5_filepath,
    used_markers=_base_.used_markers,
    patch_size=_base_.patch_size,
    ignore_annotation=_base_.ignore_annotation,
)

_base_.custom_hooks[0].type           = 'EvaluateModelRich'
_base_.custom_hooks[0].n_jobs         = 8
_base_.custom_hooks[0].epochs         = 5000
_base_.custom_hooks[0].train_indicies = _base_.train_indicies
_base_.custom_hooks[0].val_indicies   = _base_.test_indicies
_base_.custom_hooks[0].pipeline       = _base_.val_pipeline
_base_.custom_hooks[0].dataset_kwargs = dataset_kwargs
{amap_line}
# ── Model ─────────────────────────────────────────────────────────────────────
_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.neck.in_channels     = _base_.mix_channels  # 512

work_dir = '{work_dir}'
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    n_written = 0
    for ds_name, ds in DATASETS.items():
        folds = list(range(N_FOLDS)) if ds['splits'] else [None]
        out_dir = OUT_ROOT / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in MODELS.items():
            for fold in folds:
                suffix = f'_fold{fold}' if fold is not None else ''
                fname  = f'{model_name}_VICReg{suffix}.py'

                if model['lars']:
                    content = make_funnel_config(ds_name, ds, fold)
                else:
                    content = make_standard_config(ds_name, ds, model_name, model, fold)

                path = out_dir / fname
                path.write_text(content)
                print(f'  {path.relative_to(OUT_ROOT.parent.parent.parent)}')
                n_written += 1

    print(f'\n{n_written} configs written to {OUT_ROOT}')


if __name__ == '__main__':
    main()
