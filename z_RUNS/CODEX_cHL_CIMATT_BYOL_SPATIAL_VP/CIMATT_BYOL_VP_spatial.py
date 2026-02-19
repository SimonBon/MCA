backbone = dict(
    block_width=2,
    drop_prob=0.05,
    in_channels=41,
    layer_config=[
        1,
        1,
    ],
    n_heads=4,
    stem_width=32,
    type='WideModelAttention')
batch_size = 256
custom_hooks = [
    dict(
        dataset_kwargs=dict(
            h5_filepath=
            '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5',
            ignore_annotation=[
                'Seg Artifact',
            ],
            patch_size=32,
            preprocess=None,
            used_indicies=
            '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
            used_markers=
            '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
        ),
        pipeline=[
            dict(
                n_views=[
                    1,
                ],
                transforms=[
                    [
                        dict(size=24, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        short=False,
        train_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
        type='EvaluateModel',
        val_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/val.txt'
    ),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'MCA.configs._datasets_',
        'MCA.src.dataset',
        'MCA.src.transforms',
        'MCA.src.SimCLR',
        'MCA.src.VICReg',
        'MCA.src.BYOL',
        'MCA.src.models',
        'MCA.src.models_attention',
        'MCA.src.val_hook',
    ])
cutter_size = 24
dataset = dict(type='MCIDataset')
dataset_kwargs = dict(
    h5_filepath=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5',
    ignore_annotation=[
        'Seg Artifact',
    ],
    patch_size=32,
    preprocess=None,
    used_indicies=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
    used_markers=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=50, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmselfsup'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
features_per_marker = 32
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5'
ignore_annotation = [
    'Seg Artifact',
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    custom_cfg=[
        dict(data_src='', method='mean', window_size='global'),
    ],
    window_size=1)
lr = 0.0003
mask_patch = True
model = dict(
    backbone=dict(
        block_width=2,
        drop_prob=0.05,
        in_channels=41,
        layer_config=[
            1,
            1,
        ],
        n_heads=4,
        stem_width=32,
        type='WideModelAttention'),
    data_preprocessor=None,
    momentum=0.996,
    neck=dict(
        hid_channels=512,
        in_channels=1312,
        num_layers=2,
        out_channels=256,
        type='NonLinearNeck',
        with_avg_pool=False),
    pred_hid_channels=512,
    pred_in_channels=256,
    pred_out_channels=256,
    type='MVBYOL')
n_cosine = 800
n_linear = 200
n_markers = 41
num_workers = 16
optim_wrapper = dict(
    optimizer=dict(lr=0.0003, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
optimizer = dict(lr=0.0003, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=200, start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=800,
        begin=200,
        by_epoch=False,
        end=1000,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
patch_size = 32
preprocess = None
resume = False
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/test.txt'
train_aug_strong = [
    dict(horizontal=True, prob=0.5, type='C_RandomFlip', vertical=True),
    dict(
        angle=(
            0,
            360,
        ),
        order=1,
        scale=(
            0.66,
            1.5,
        ),
        shift=(
            -0.1,
            0.1,
        ),
        type='C_RandomAffine'),
    dict(
        clip=True,
        scale=(
            0.95,
            1.05,
        ),
        shift=(
            -0.01,
            0.01,
        ),
        type='C_RandomChannelShiftScale'),
    dict(
        clip=True, strength=(
            -0.15,
            0.15,
        ), type='C_RandomBackgroundGradient'),
    dict(clip=True, mean=(
        0,
        0,
    ), std=(
        0,
        0.05,
    ), type='C_RandomNoise'),
    dict(size=24, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
train_aug_weak = [
    dict(horizontal=True, prob=0.3, type='C_RandomFlip', vertical=True),
    dict(
        angle=(
            0,
            360,
        ),
        order=1,
        scale=(
            0.9,
            1.1,
        ),
        shift=(
            0,
            0,
        ),
        type='C_RandomAffine'),
    dict(
        clip=True,
        scale=(
            0.98,
            1.02,
        ),
        shift=(
            -0.005,
            0.005,
        ),
        type='C_RandomChannelShiftScale'),
    dict(clip=True, strength=(
        0.0,
        0.05,
    ), type='C_RandomBackgroundGradient'),
    dict(clip=True, mean=(
        0,
        0,
    ), std=(
        0,
        0.02,
    ), type='C_RandomNoise'),
    dict(size=24, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
train_cfg = dict(max_iters=1000, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_filepath=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5',
        ignore_annotation=[
            'Seg Artifact',
        ],
        mask_patch=True,
        patch_size=32,
        pipeline=[
            dict(
                n_views=[
                    1,
                    1,
                ],
                transforms=[
                    [
                        dict(
                            horizontal=True,
                            prob=0.5,
                            type='C_RandomFlip',
                            vertical=True),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.66,
                                1.5,
                            ),
                            shift=(
                                -0.1,
                                0.1,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            scale=(
                                0.95,
                                1.05,
                            ),
                            shift=(
                                -0.01,
                                0.01,
                            ),
                            type='C_RandomChannelShiftScale'),
                        dict(
                            clip=True,
                            strength=(
                                -0.15,
                                0.15,
                            ),
                            type='C_RandomBackgroundGradient'),
                        dict(
                            clip=True,
                            mean=(
                                0,
                                0,
                            ),
                            std=(
                                0,
                                0.05,
                            ),
                            type='C_RandomNoise'),
                        dict(size=24, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                    [
                        dict(
                            horizontal=True,
                            prob=0.3,
                            type='C_RandomFlip',
                            vertical=True),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.9,
                                1.1,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            scale=(
                                0.98,
                                1.02,
                            ),
                            shift=(
                                -0.005,
                                0.005,
                            ),
                            type='C_RandomChannelShiftScale'),
                        dict(
                            clip=True,
                            strength=(
                                0.0,
                                0.05,
                            ),
                            type='C_RandomBackgroundGradient'),
                        dict(
                            clip=True,
                            mean=(
                                0,
                                0,
                            ),
                            std=(
                                0,
                                0.02,
                            ),
                            type='C_RandomNoise'),
                        dict(size=24, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        preprocess=None,
        type='MCIDataset',
        used_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
        used_markers=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
    ),
    drop_last=True,
    num_workers=16,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_dataset = dict(
    h5_filepath=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5',
    ignore_annotation=[
        'Seg Artifact',
    ],
    mask_patch=True,
    patch_size=32,
    pipeline=[
        dict(
            n_views=[
                1,
                1,
            ],
            transforms=[
                [
                    dict(
                        horizontal=True,
                        prob=0.5,
                        type='C_RandomFlip',
                        vertical=True),
                    dict(
                        angle=(
                            0,
                            360,
                        ),
                        order=1,
                        scale=(
                            0.66,
                            1.5,
                        ),
                        shift=(
                            -0.1,
                            0.1,
                        ),
                        type='C_RandomAffine'),
                    dict(
                        clip=True,
                        scale=(
                            0.95,
                            1.05,
                        ),
                        shift=(
                            -0.01,
                            0.01,
                        ),
                        type='C_RandomChannelShiftScale'),
                    dict(
                        clip=True,
                        strength=(
                            -0.15,
                            0.15,
                        ),
                        type='C_RandomBackgroundGradient'),
                    dict(
                        clip=True,
                        mean=(
                            0,
                            0,
                        ),
                        std=(
                            0,
                            0.05,
                        ),
                        type='C_RandomNoise'),
                    dict(size=24, type='C_CentralCutter'),
                    dict(type='C_ToTensor'),
                ],
                [
                    dict(
                        horizontal=True,
                        prob=0.3,
                        type='C_RandomFlip',
                        vertical=True),
                    dict(
                        angle=(
                            0,
                            360,
                        ),
                        order=1,
                        scale=(
                            0.9,
                            1.1,
                        ),
                        shift=(
                            0,
                            0,
                        ),
                        type='C_RandomAffine'),
                    dict(
                        clip=True,
                        scale=(
                            0.98,
                            1.02,
                        ),
                        shift=(
                            -0.005,
                            0.005,
                        ),
                        type='C_RandomChannelShiftScale'),
                    dict(
                        clip=True,
                        strength=(
                            0.0,
                            0.05,
                        ),
                        type='C_RandomBackgroundGradient'),
                    dict(
                        clip=True,
                        mean=(
                            0,
                            0,
                        ),
                        std=(
                            0,
                            0.02,
                        ),
                        type='C_RandomNoise'),
                    dict(size=24, type='C_CentralCutter'),
                    dict(type='C_ToTensor'),
                ],
            ],
            type='C_MultiView'),
        dict(type='C_PackInputs'),
    ],
    preprocess=None,
    type='MCIDataset',
    used_indicies=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
    used_markers=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
)
train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt'
train_pipeline = [
    dict(
        n_views=[
            1,
            1,
        ],
        transforms=[
            [
                dict(
                    horizontal=True,
                    prob=0.5,
                    type='C_RandomFlip',
                    vertical=True),
                dict(
                    angle=(
                        0,
                        360,
                    ),
                    order=1,
                    scale=(
                        0.66,
                        1.5,
                    ),
                    shift=(
                        -0.1,
                        0.1,
                    ),
                    type='C_RandomAffine'),
                dict(
                    clip=True,
                    scale=(
                        0.95,
                        1.05,
                    ),
                    shift=(
                        -0.01,
                        0.01,
                    ),
                    type='C_RandomChannelShiftScale'),
                dict(
                    clip=True,
                    strength=(
                        -0.15,
                        0.15,
                    ),
                    type='C_RandomBackgroundGradient'),
                dict(
                    clip=True,
                    mean=(
                        0,
                        0,
                    ),
                    std=(
                        0,
                        0.05,
                    ),
                    type='C_RandomNoise'),
                dict(size=24, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
            [
                dict(
                    horizontal=True,
                    prob=0.3,
                    type='C_RandomFlip',
                    vertical=True),
                dict(
                    angle=(
                        0,
                        360,
                    ),
                    order=1,
                    scale=(
                        0.9,
                        1.1,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(
                    clip=True,
                    scale=(
                        0.98,
                        1.02,
                    ),
                    shift=(
                        -0.005,
                        0.005,
                    ),
                    type='C_RandomChannelShiftScale'),
                dict(
                    clip=True,
                    strength=(
                        0.0,
                        0.05,
                    ),
                    type='C_RandomBackgroundGradient'),
                dict(
                    clip=True,
                    mean=(
                        0,
                        0,
                    ),
                    std=(
                        0,
                        0.02,
                    ),
                    type='C_RandomNoise'),
                dict(size=24, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
        ],
        type='C_MultiView'),
    dict(type='C_PackInputs'),
]
used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
val_augmentation = [
    dict(size=24, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/val.txt'
val_pipeline = [
    dict(
        n_views=[
            1,
        ],
        transforms=[
            [
                dict(size=24, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
        ],
        type='C_MultiView'),
    dict(type='C_PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SelfSupVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS/CODEX_cHL_CIMATT_BYOL_SPATIAL_VP'
