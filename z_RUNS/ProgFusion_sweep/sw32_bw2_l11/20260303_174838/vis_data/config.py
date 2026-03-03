backbone = dict(
    block_width=2,
    drop_prob=0.05,
    in_channels=31,
    input_norm=True,
    layer_config=[
        1,
        1,
    ],
    stem_width=32,
    type='WideModelProgressiveFusion')
batch_size = 512
custom_hooks = [
    dict(
        dataset_kwargs=dict(
            h5_filepath=
            '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5',
            ignore_annotation=[
                'Other',
                'NK_DC',
            ],
            patch_size=24,
            used_markers=
            '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
        ),
        epochs=500,
        le_fractions=[],
        le_n_per_class=[],
        max_samples=15000,
        n_jobs=8,
        pipeline=[
            dict(
                n_views=[
                    1,
                ],
                transforms=[
                    [
                        dict(size=10, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        train_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/train.txt',
        type='EvaluateModelRich',
        val_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/val.txt'
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
cutter_size = 12
dataset = dict(
    h5_filepath=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5',
    ignore_annotation=[
        'Other',
        'NK_DC',
    ],
    patch_size=24,
    type='MCIDataset',
    used_markers=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
)
dataset_kwargs = dict(
    h5_filepath=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5',
    ignore_annotation=[
        'Other',
        'NK_DC',
    ],
    patch_size=24,
    used_markers=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
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
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5'
ignore_annotation = [
    'Other',
    'NK_DC',
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
model = dict(
    backbone=dict(
        block_width=2,
        drop_prob=0.05,
        in_channels=31,
        input_norm=True,
        layer_config=[
            1,
            1,
        ],
        stem_width=32,
        type='WideModelProgressiveFusion'),
    cov_coeff=1.0,
    data_preprocessor=None,
    gamma=1.0,
    neck=dict(
        hid_channels=512,
        in_channels=992,
        num_layers=2,
        out_channels=512,
        type='NonLinearNeck',
        with_avg_pool=False),
    sim_coeff=25.0,
    std_coeff=25.0,
    type='MVVICReg')
n_cosine = 450
n_linear = 50
n_markers = 31
num_workers = 8
optim_wrapper = dict(
    optimizer=dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=50, start_factor=0.0001, type='LinearLR'),
    dict(
        T_max=450,
        begin=50,
        by_epoch=False,
        end=500,
        eta_min=0.03,
        type='CosineAnnealingLR'),
]
patch_size = 24
resume = False
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/test.txt'
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
    dict(size=12, type='C_CentralCutter'),
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
    dict(size=10, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
train_cfg = dict(max_iters=500, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=512,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_filepath=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5',
        ignore_annotation=[
            'Other',
            'NK_DC',
        ],
        mask_patch=True,
        patch_size=24,
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
                        dict(size=12, type='C_CentralCutter'),
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
                        dict(size=10, type='C_CentralCutter'),
                        dict(type='C_ToTensor'),
                    ],
                ],
                type='C_MultiView'),
            dict(type='C_PackInputs'),
        ],
        type='MCIDataset',
        used_indicies=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/train.txt',
        used_markers=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
    ),
    drop_last=True,
    num_workers=8,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_dataset = dict(
    h5_filepath=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5',
    ignore_annotation=[
        'Other',
        'NK_DC',
    ],
    mask_patch=True,
    patch_size=24,
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
                    dict(size=12, type='C_CentralCutter'),
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
                    dict(size=10, type='C_CentralCutter'),
                    dict(type='C_ToTensor'),
                ],
            ],
            type='C_MultiView'),
        dict(type='C_PackInputs'),
    ],
    type='MCIDataset',
    used_indicies=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/train.txt',
    used_markers=
    '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
)
train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/train.txt'
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
                dict(size=12, type='C_CentralCutter'),
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
                dict(size=10, type='C_CentralCutter'),
                dict(type='C_ToTensor'),
            ],
        ],
        type='C_MultiView'),
    dict(type='C_PackInputs'),
]
used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
val_augmentation = [
    dict(size=10, type='C_CentralCutter'),
    dict(type='C_ToTensor'),
]
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/val.txt'
val_pipeline = [
    dict(
        n_views=[
            1,
        ],
        transforms=[
            [
                dict(size=10, type='C_CentralCutter'),
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
work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS/ProgFusion_sweep/sw32_bw2_l11'
