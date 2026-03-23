from copy import deepcopy

_base_ = [
    '../../../_base_/default.py',
    '../../../_augmentations_/high.py',
    '../../../_base_/train_cfg.py',
    '../../../_base_/val_cfg.py',
    '../../../_datasets_/MIBI_TNBC.py',
    '../../../_backbones_/CIM_LateFusion_MLP.py',
    '../../../_algorithms_/VICReg.py',
]

batch_size  = 256
num_workers = 16
mask_patch  = True

# ── Override paths to nobackup ──────────────────────────────────────────────
_base_.h5_filepath    = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/MIBI_TNBC.h5'
_base_.used_markers   = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/used_markers.txt'
_base_.train_indicies = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/train.txt'
_base_.test_indicies  = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/MIBI_TNBC/test.txt'
_base_.dataset['h5_filepath']  = _base_.h5_filepath
_base_.dataset['used_markers'] = _base_.used_markers

_base_.val_augmentation[0].size = _base_.cutter_size
_base_.val_pipeline[0].transforms = [_base_.val_augmentation]

_base_.train_aug_strong[-2].size = _base_.cutter_size
_base_.train_aug_weak[-2].size   = _base_.cutter_size
_base_.train_pipeline[0].transforms = [_base_.train_aug_strong, _base_.train_aug_weak]

train_dataset = deepcopy(_base_.dataset)
train_dataset['used_indicies'] = _base_.train_indicies
train_dataset['pipeline']      = _base_.train_pipeline
train_dataset['mask_patch']    = mask_patch

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

_base_.model.backbone             = _base_.backbone
_base_.model.backbone.in_channels = _base_.n_markers
_base_.model.neck.in_channels     = _base_.n_markers * _base_.features_per_marker

work_dir = '/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/paper_clean/MIBI_TNBC/CIM_LateFusion'
