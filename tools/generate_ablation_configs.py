#!/usr/bin/env python3
"""Generate CIM_Funnel_Large ablation configs for CODEX_cHL.

Ablation groups
---------------
1. Channel augmentation  — what do domain-specific channel augs contribute?
2. Training length       — how many iters does the Funnel need?
3. Model capacity        — mix_n_blocks and mix_channels

All runs: CODEX_cHL single split, LARS optimiser.
Standard ablation length: 8 000 iters (n_linear=400, n_cosine=7600).
16 000-iter paper baseline is already in paper_clean/.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
NB      = '/nobackup/lab_taschner-mandl/simongutwein'
DATA    = f'{NB}/h5_files/CODEX_cHL'
RUNS    = f'{NB}/z_RUNS/ablations/CIM_Funnel'
OUT_DIR = Path(__file__).parent.parent / 'configs' / '_experiments_' / 'ablations' / 'CIM_Funnel'

# ── Fixed paths / dataset config ───────────────────────────────────────────────
PATH_BLOCK = f"""\
_base_.h5_filepath    = '{DATA}/CODEX_cHL.h5'
_base_.used_markers   = '{DATA}/used_markers.txt'
_base_.train_indicies = '{DATA}/train.txt'
_base_.test_indicies  = '{DATA}/test.txt'
_base_.dataset_kwargs['h5_filepath']  = _base_.h5_filepath
_base_.dataset_kwargs['used_markers'] = _base_.used_markers"""

EVAL_BLOCK = f"""\
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
_base_.custom_hooks[0].annotation_map = {{'Cytotoxic CD8': 'CD8', 'TReg': 'Treg'}}"""

FULL_AUG_STRONG = """\
train_aug_strong = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size=24),
    dict(type='C_ToTensor'),
]"""

FULL_AUG_WEAK = """\
train_aug_weak = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size=22),
    dict(type='C_ToTensor'),
]"""

PIPELINE = """\
train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[train_aug_strong, train_aug_weak]),
    dict(type='C_PackInputs'),
]

_base_.val_augmentation[0].size = 22
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]"""

DATASET_BLOCK = """\
train_dataset = deepcopy(_base_.dataset)
train_dataset.update(_base_.dataset_kwargs)
train_dataset['used_indicies'] = _base_.train_indicies
train_dataset['pipeline']      = train_pipeline
train_dataset['mask_patch']    = mask_patch

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=train_dataset,
)"""


def scheduler_block(n_iters):
    n_linear = max(200, n_iters // 20)   # 5 % warmup
    n_cosine = n_iters - n_linear
    return f"""\
n_linear      = {n_linear}
n_cosine      = {n_cosine}
optimizer     = dict(type='LARS', lr=0.3, momentum=0.9, weight_decay=1e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_cfg     = dict(type='IterBasedTrainLoop', max_iters=n_linear + n_cosine)
param_scheduler = [
    dict(type='LinearLR',          start_factor=1e-4, by_epoch=False, begin=0,        end=n_linear),
    dict(type='CosineAnnealingLR', T_max=n_cosine,    by_epoch=False, begin=n_linear, end=n_linear + n_cosine, eta_min=0.03),
]"""


def model_block(mix_channels, mix_n_blocks):
    return f"""\
_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.backbone.mix_channels = {mix_channels}
_base_.model.backbone.mix_n_blocks = {mix_n_blocks}
_base_.model.neck.in_channels     = {mix_channels}"""


def make_config(name, aug_strong, aug_weak, n_iters=8000,
                mix_channels=512, mix_n_blocks=8):
    work_dir = f'{RUNS}/{name}'
    return f"""\
from copy import deepcopy

_base_ = [
    '../../../_base_/default.py',
    '../../../_base_/val_cfg.py',
    '../../../_datasets_/CODEX_cHL.py',
    '../../../_backbones_/CIM_Funnel_Large.py',
    '../../../_algorithms_/VICReg.py',
]

mask_patch = True

{PATH_BLOCK}

# ── Optimizer / schedule ──────────────────────────────────────────────────────
{scheduler_block(n_iters)}

# ── Augmentations ─────────────────────────────────────────────────────────────
{aug_strong}

{aug_weak}

{PIPELINE}

# ── Dataset ───────────────────────────────────────────────────────────────────
{DATASET_BLOCK}

# ── Eval hook ─────────────────────────────────────────────────────────────────
{EVAL_BLOCK}

# ── Model ─────────────────────────────────────────────────────────────────────
{model_block(mix_channels, mix_n_blocks)}

work_dir = '{work_dir}'
"""


# ── Augmentation variants ──────────────────────────────────────────────────────

def aug_strong_variant(drop_prob=0.1, channel_shift=True, noise=True, affine=True):
    transforms = [
        "    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),",
    ]
    if affine:
        transforms.append(
            "    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),")
    if channel_shift:
        transforms.append(
            "    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),")
    if noise:
        transforms.append(
            "    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),")
    if drop_prob > 0:
        transforms.append(
            f"    dict(type='C_RandomChannelDrop',       drop_prob={drop_prob}),")
    transforms += [
        "    dict(type='C_CentralCutter',           size=24),",
        "    dict(type='C_ToTensor'),",
    ]
    body = "\n".join(transforms)
    return f"train_aug_strong = [\n{body}\n]"


def aug_weak_variant(drop_prob=0.1, channel_shift=True, noise=True, affine=True):
    transforms = [
        "    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),",
    ]
    if affine:
        transforms.append(
            "    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),")
    if channel_shift:
        transforms.append(
            "    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),")
    if noise:
        transforms.append(
            "    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),")
    if drop_prob > 0:
        transforms.append(
            f"    dict(type='C_RandomChannelDrop',       drop_prob={drop_prob}),")
    transforms += [
        "    dict(type='C_CentralCutter',           size=22),",
        "    dict(type='C_ToTensor'),",
    ]
    body = "\n".join(transforms)
    return f"train_aug_weak = [\n{body}\n]"


# ── Define all ablations ───────────────────────────────────────────────────────

ABLATIONS = []

# --- Group 1: Channel augmentation ---
# Baseline at 8k (all channel augs on)
ABLATIONS.append(dict(
    name='aug_full_8k',
    aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=True,  noise=True,  affine=True),
    aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=True,  noise=True,  affine=True),
    n_iters=8000,
))

# No channel augmentation at all (only spatial: flip + affine + crop)
ABLATIONS.append(dict(
    name='aug_no_channel',
    aug_strong=aug_strong_variant(drop_prob=0.0, channel_shift=False, noise=False, affine=True),
    aug_weak  =aug_weak_variant(  drop_prob=0.0, channel_shift=False, noise=False, affine=True),
    n_iters=8000,
))

# No channel drop only
ABLATIONS.append(dict(
    name='aug_no_drop',
    aug_strong=aug_strong_variant(drop_prob=0.0, channel_shift=True,  noise=True,  affine=True),
    aug_weak  =aug_weak_variant(  drop_prob=0.0, channel_shift=True,  noise=True,  affine=True),
    n_iters=8000,
))

# No channel shift/scale only
ABLATIONS.append(dict(
    name='aug_no_shift',
    aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=False, noise=True,  affine=True),
    aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=False, noise=True,  affine=True),
    n_iters=8000,
))

# No noise only
ABLATIONS.append(dict(
    name='aug_no_noise',
    aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=True,  noise=False, affine=True),
    aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=True,  noise=False, affine=True),
    n_iters=8000,
))

# Channel drop probability sweep
for dp in [0.05, 0.2]:
    ABLATIONS.append(dict(
        name=f'aug_drop_{str(dp).replace(".", "")}',
        aug_strong=aug_strong_variant(drop_prob=dp, channel_shift=True, noise=True, affine=True),
        aug_weak  =aug_weak_variant(  drop_prob=dp, channel_shift=True, noise=True, affine=True),
        n_iters=8000,
    ))

# --- Group 2: Training length ---
for n in [2000, 4000]:
    ABLATIONS.append(dict(
        name=f'iters_{n//1000}k',
        aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        n_iters=n,
    ))
# 8k covered by aug_full_8k; 16k is the paper baseline

# --- Group 3: Model capacity ---
# Vary mix_n_blocks
for n_blocks in [4, 12]:
    ABLATIONS.append(dict(
        name=f'cap_blocks_{n_blocks}',
        aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        n_iters=8000,
        mix_n_blocks=n_blocks,
    ))

# Vary mix_channels
for ch in [256, 768]:
    ABLATIONS.append(dict(
        name=f'cap_ch_{ch}',
        aug_strong=aug_strong_variant(drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        aug_weak  =aug_weak_variant(  drop_prob=0.1, channel_shift=True, noise=True, affine=True),
        n_iters=8000,
        mix_channels=ch,
    ))


# ── Write configs ──────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for abl in ABLATIONS:
        name       = abl['name']
        n_iters    = abl.get('n_iters', 8000)
        mix_ch     = abl.get('mix_channels', 512)
        mix_blocks = abl.get('mix_n_blocks', 8)
        content    = make_config(
            name       = name,
            aug_strong = abl['aug_strong'],
            aug_weak   = abl['aug_weak'],
            n_iters    = n_iters,
            mix_channels  = mix_ch,
            mix_n_blocks  = mix_blocks,
        )
        path = OUT_DIR / f'{name}.py'
        path.write_text(content)
        print(f'  {path.relative_to(OUT_DIR.parent.parent.parent)}')

    print(f'\n{len(ABLATIONS)} configs written to {OUT_DIR}')
    print('\nGroups:')
    print('  aug_*      — channel augmentation ablations (8k iters)')
    print('  iters_*    — training length ablations')
    print('  cap_*      — model capacity ablations (8k iters)')
    print('\nReference:')
    print('  aug_full_8k  = 8k-iter baseline for ablation comparisons')
    print('  paper_clean/CODEX_cHL/CIM_Funnel_Large = 16k full paper run')


if __name__ == '__main__':
    main()
