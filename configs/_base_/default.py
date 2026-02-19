custom_imports = dict(
    imports=[
        'MCA.configs._datasets_',
        'MCA.src.dataset',
        'MCA.src.transforms',
        'MCA.src.SimCLR',
        'MCA.src.VICReg',
        'MCA.src.BYOL',
        'MCA.src.models',
        'MCA.src.models_attention',
        'MCA.src.val_hook'
        ],
    allow_failed_imports=False,
)

default_scope = 'mmselfsup'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=3),
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

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