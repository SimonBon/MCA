from copy import deepcopy

_base_ = [
    '../../../_base_/default.py',
    '../../../_base_/val_cfg.py',
    '../../../_datasets_/MIBI_TNBC.py',
    '../../../_backbones_/CIM_Funnel_Large.py',
    '../../../_algorithms_/VICReg.py',
]

batch_size  = 128
num_workers = 16
mask_patch  = True

# ── Override paths to nobackup ──────────────────────────────────────────────
_base_.h5_filepath    = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5'
_base_.used_markers   = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt'
_base_.train_indicies = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/cv_splits_paper/split_0/train.txt'
_base_.test_indicies  = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/cv_splits_paper/split_0/test.txt'
_base_.dataset['h5_filepath']  = _base_.h5_filepath
_base_.dataset['used_markers'] = _base_.used_markers

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
    dict(type='C_CentralCutter',           size=20),
    dict(type='C_ToTensor'),
]

train_aug_weak = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size=18),
    dict(type='C_ToTensor'),
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[train_aug_strong, train_aug_weak]),
    dict(type='C_PackInputs'),
]

_base_.val_augmentation[0].size = 18
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]

# ── Dataset ───────────────────────────────────────────────────────────────────
train_dataset = deepcopy(_base_.dataset)
train_dataset['used_indicies'] = _base_.train_indicies
train_dataset['pipeline']      = train_pipeline
train_dataset['mask_patch']    = mask_patch

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

# ── Model ─────────────────────────────────────────────────────────────────────
_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.neck.in_channels     = _base_.mix_channels  # 512

work_dir = '/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/MIBI_TNBC/CIM_Funnel_Large/fold_0'
