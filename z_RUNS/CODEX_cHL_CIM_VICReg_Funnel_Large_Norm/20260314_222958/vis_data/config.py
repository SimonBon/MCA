_DATA = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL'
backbone = dict(
    block_width=4,
    drop_prob=0.05,
    in_channels=41,
    input_norm=True,
    mix_channels=512,
    mix_n_blocks=8,
    sep_layer_config=[
        2,
        2,
    ],
    stem_width=16,
    type='CIM_Funnel')
batch_size = 256
custom_hooks = [
    dict(
        dataset_kwargs=dict(
            h5_filepath=
            '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5',
            ignore_annotation=[
                'Seg Artifact',
            ],
            patch_size=32,
            used_markers=
            '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt'
        ),
        epochs=5000,
        max_samples=None,
        n_jobs=8,
        pipeline=[
            dict(
                n_views=[
                    1,
                ],
                transforms=[
                    [
                        dict(size=22, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        train_indicies=
        '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/train.txt',
        type='EvaluateModelRich',
        val_indicies=
        '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/val.txt'
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
        'MCA.src.models_early_fusion',
        'MCA.src.MCM',
        'MCA.src.val_hook',
        'MCA.src.val_hook_rich',
    ])
cutter_size = 24
dataset = dict(type='MCIDataset')
dataset_kwargs = dict(
    h5_filepath=
    '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5',
    ignore_annotation=[
        'Seg Artifact',
    ],
    patch_size=32,
    preprocess=None,
    used_indicies=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt',
    used_markers=
    '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt'
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
h5_filepath = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5'
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
mask_patch = True
mix_channels = 512
model = dict(
    backbone=dict(
        block_width=4,
        drop_prob=0.05,
        in_channels=41,
        input_norm=True,
        mix_channels=512,
        mix_n_blocks=8,
        sep_layer_config=[
            2,
            2,
        ],
        stem_width=16,
        type='CIM_Funnel'),
    cov_coeff=1.0,
    data_preprocessor=None,
    gamma=1.0,
    neck=dict(
        hid_channels=512,
        in_channels=512,
        num_layers=2,
        out_channels=512,
        type='NonLinearNeck',
        with_avg_pool=False),
    sim_coeff=25.0,
    std_coeff=25.0,
    type='MVVICReg')
n_cosine = 15200
n_linear = 800
n_markers = 41
num_workers = 16
optim_wrapper = dict(
    optimizer=dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=800, start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=15200,
        begin=800,
        by_epoch=False,
        end=16000,
        eta_min=0.03,
        type='CosineAnnealingLR'),
]
patch_size = 32
preprocess = None
resume = False
test_indicies = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/test.txt'
train_aug_strong = [
    dict(horizontal=True, prob=0.5, type='C_RandomFlip', vertical=True),
    dict(
        angle=(
            0,
            360,
        ),
        order=1,
        scale=(
            0.8,
            1.2,
        ),
        shift=(
            0,
            0,
        ),
        type='C_RandomAffine'),
    dict(
        clip=True,
        scale=(
            0.9,
            1.2,
        ),
        shift=(
            0,
            0,
        ),
        type='C_RandomChannelShiftScale'),
    dict(clip=True, mean=(
        0,
        0,
    ), std=(
        0,
        0.02,
    ), type='C_RandomNoise'),
    dict(drop_prob=0.1, type='C_RandomChannelDrop'),
    dict(size=24, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
train_aug_weak = [
    dict(horizontal=True, prob=0.5, type='C_RandomFlip', vertical=True),
    dict(
        angle=(
            0,
            360,
        ),
        order=1,
        scale=(
            0.8,
            1.2,
        ),
        shift=(
            0,
            0,
        ),
        type='C_RandomAffine'),
    dict(
        clip=True,
        scale=(
            0.9,
            1.2,
        ),
        shift=(
            0,
            0,
        ),
        type='C_RandomChannelShiftScale'),
    dict(clip=True, mean=(
        0,
        0,
    ), std=(
        0,
        0.02,
    ), type='C_RandomNoise'),
    dict(drop_prob=0.1, type='C_RandomChannelDrop'),
    dict(size=22, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
train_cfg = dict(max_iters=16000, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_filepath=
        '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5',
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
                                0.8,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            scale=(
                                0.9,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomChannelShiftScale'),
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
                        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                        dict(size=24, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
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
                                0.8,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            scale=(
                                0.9,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomChannelShiftScale'),
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
                        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                        dict(size=22, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        preprocess=None,
        type='MCIDataset',
        used_indicies=
        '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/train.txt',
        used_markers=
        '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt'
    ),
    drop_last=True,
    num_workers=16,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_dataset = dict(
    h5_filepath=
    '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/CODEX_cHL.h5',
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
                            0.8,
                            1.2,
                        ),
                        shift=(
                            0,
                            0,
                        ),
                        type='C_RandomAffine'),
                    dict(
                        clip=True,
                        scale=(
                            0.9,
                            1.2,
                        ),
                        shift=(
                            0,
                            0,
                        ),
                        type='C_RandomChannelShiftScale'),
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
                    dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                    dict(size=24, type='C_CentralCutter'),
                    dict(type='C_ToTensor'),
                ],
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
                            0.8,
                            1.2,
                        ),
                        shift=(
                            0,
                            0,
                        ),
                        type='C_RandomAffine'),
                    dict(
                        clip=True,
                        scale=(
                            0.9,
                            1.2,
                        ),
                        shift=(
                            0,
                            0,
                        ),
                        type='C_RandomChannelShiftScale'),
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
                    dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                    dict(size=22, type='C_CentralCutter'),
                    dict(type='C_ToTensor'),
                ],
            ],
            type='C_MultiView'),
        dict(type='C_PackInputs'),
    ],
    preprocess=None,
    type='MCIDataset',
    used_indicies=
    '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/train.txt',
    used_markers=
    '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt'
)
train_indicies = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/train.txt'
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
                        0.8,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(
                    clip=True,
                    scale=(
                        0.9,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomChannelShiftScale'),
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
                dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                dict(size=24, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
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
                        0.8,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(
                    clip=True,
                    scale=(
                        0.9,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomChannelShiftScale'),
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
                dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                dict(size=22, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
        ],
        type='C_MultiView'),
    dict(type='C_PackInputs'),
]
used_markers = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/used_markers.txt'
val_augmentation = [
    dict(size=22, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
val_indicies = '/nobackup/lab_taschner-mandl/simongutwein/h5_files/CODEX_cHL/val.txt'
val_pipeline = [
    dict(
        n_views=[
            1,
        ],
        transforms=[
            [
                dict(size=22, type='C_CentralCutter'),
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
work_dir = '/nobackup/lab_taschner-mandl/simongutwein/z_RUNS/CODEX_cHL_CIM_VICReg_Funnel_Large_Norm'
