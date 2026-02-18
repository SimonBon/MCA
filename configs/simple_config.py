custom_imports = dict(
    imports=[
        'MCA.SimCLR',
        'MCA.dataset',
        'MCA.transforms',
        'MCA.models'
    ],
    allow_failed_imports=False,
)

default_scope = 'mmselfsup'

h5_file_path = "/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/dataset/cHL_CODEX.h5"

markers_to_use = None #['DAPI-01', 'CD11b', 'CD11c', 'CD15', 'CD163', 'CD20', 'CD206', 'CD30', 'CD31', 'CD4', 'CD56', 'CD68', 'CD7', 'CD8', 'Cytokeritin', 'MCT', 'Podoplanin']
crops = [20, 20]
# val_crop = min(crops)
image_key = 'IMAGES'
mask_image = True
in_channels = 49 #len(markers_to_use)
stem_width=32
block_width=2
feature_dim = in_channels*stem_width
batch_size = 512
classes_to_ignore=['Seg Artifact']

view_pipelines = [
    [
        dict(type='C_RandomFlip', prob=0.5, horizontal=True, vertical=True),
        dict(type='C_RandomAffine', angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
        dict(type='RandomIntensity', low=2/3, high=3/2, clip=True),
        dict(type='RandomNoise', mean=(0, 0), std=(0, 0.03), clip=True),
        dict(type='C_RandomChannelMixup', mixup_prob=0.1),
        dict(type='C_RandomChannelDrop', drop_prob=0.1),
        dict(type='CentralCutter', size=crop),
    ] for crop in crops
]

# val_pipeline = [dict(type='CentralCutter', size=val_crop)]

train_pipeline = [
    dict(type='MultiView', num_views=[1] * len(crops), transforms=view_pipelines),
    dict(type='PackSelfSupInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=32, 
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=dict(
        type='MultiChannelDataset',
        patch_size=int(max(crops)*2),
        h5_file_path=h5_file_path,
        pipeline=train_pipeline,
        in_memory=False,
        split=[0.7, 0.1, 0.2],
        used_split='training',
        mask_image=mask_image,
        image_key=image_key,
        classes_to_ignore=classes_to_ignore,
        markers_to_use=markers_to_use
    ),
)

data_preprocessor = None

model = dict(
    type='MVSimCLR',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='WideModel',
        in_channels=in_channels, 
        stem_width=stem_width, 
        block_width=block_width, 
        layer_config=[1,1], 
        late_fusion=False
    ),
    neck=dict(
        type='NonLinearNeck',   
        in_channels=feature_dim,
        hid_channels=128,
        out_channels=128,
        num_layers=2,
        with_avg_pool=False,
    ),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2,     
    )
)

# =========================================================
# Optimizer
# =========================================================
lr = 0.3  # higher LR for contrastive, batch=1024
# optimizer = dict(type='Adam', lr=lr, weight_decay=1e-5)
optimizer = dict(type='LARS', lr=lr, momentum=0.9, weight_decay=1e-5)
# optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
)

# =========================================================
# Learning rate scheduler
# =========================================================
n_linear = 100    # longer warmup
n_cosine = 900

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=n_linear,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=n_cosine,
        eta_min=0.1 * lr,
        by_epoch=False,
        begin=n_linear,
        end=n_linear + n_cosine,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=n_linear + n_cosine
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=3),
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# custom_hooks = [
#     dict(type='EvaluateModel', priority='VERY_LOW', epochs=20),
# ]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=1,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')],
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer',
)

log_level = 'INFO'
load_from = None
resume = False

work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/z_RUNS'