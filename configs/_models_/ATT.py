features_per_marker = 32

model = dict(
    type='MVSimCLR',
    data_preprocessor=None,
    backbone=dict(
        type='MCIANet',
        in_channels=None, 
        input_size=None, 
        stem_dim=features_per_marker, 
        n_heads=4, 
        stem_blocks=2,
        stage1_blocks=2,
        stage2_blocks=2,
        expansion=2,
        drop_prob=0.05,
        sigma_fraction=0.35,
        spatial_init_mode='ones'
    ),
    neck=dict(
        type='NonLinearNeck',   
        in_channels=None,
        hid_channels=256,
        out_channels=256,
        num_layers=2,
        with_avg_pool=False,
    ),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2,     
    )
)