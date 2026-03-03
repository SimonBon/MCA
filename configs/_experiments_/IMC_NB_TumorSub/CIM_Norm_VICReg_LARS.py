from copy import deepcopy

_base_ = [
    '../../_base_/default.py',
    '../../_base_/val_cfg.py',
    '../../_datasets_/IMC_NB_TumorSub.py',
    '../../_backbones_/CIM_Norm.py',
    '../../_algorithms_/VICReg.py',
]

batch_size  = 512
num_workers = 16
mask_patch  = True

# ── Optimiser: LARS ──────────────────────────────────────────────────────────
n_linear = 400
n_cosine = 3600   # 4000 total

optimizer     = dict(type='LARS', lr=0.3, momentum=0.9, weight_decay=1e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=n_linear + n_cosine)

param_scheduler = [
    dict(type='LinearLR',          start_factor=1e-4, by_epoch=False, begin=0,        end=n_linear),
    dict(type='CosineAnnealingLR', T_max=n_cosine,    by_epoch=False, begin=n_linear, end=n_linear + n_cosine, eta_min=0.03),
]

# ── Augmentations ─────────────────────────────────────────────────────────────
# IMC_NB_TumorSub: patch_size=24, cutter_size=12 → strong=12, weak=10, val=10
train_aug_strong = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size=12),
    dict(type='C_ToTensor'),
]

train_aug_weak = [
    dict(type='C_RandomFlip',              prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine',            angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.2), shift=(0, 0), clip=True),
    dict(type='C_RandomNoise',             mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelDrop',       drop_prob=0.1),
    dict(type='C_CentralCutter',           size=10),
    dict(type='C_ToTensor'),
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[train_aug_strong, train_aug_weak]),
    dict(type='C_PackInputs'),
]

_base_.val_augmentation[0].size = 10
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
_base_.custom_hooks[0].train_indicies = _base_.train_indicies
_base_.custom_hooks[0].val_indicies   = _base_.val_indicies
_base_.custom_hooks[0].pipeline       = _base_.val_pipeline
_base_.custom_hooks[0].dataset_kwargs = dataset_kwargs

# ── Model ─────────────────────────────────────────────────────────────────────
_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.neck.in_channels     = _base_.n_markers * _base_.features_per_marker

work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS/IMC_NB_TumorSub_CIM_Norm_VICReg_LARS'
