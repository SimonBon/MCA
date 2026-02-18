features_per_marker = 32

model = dict(
    type='MVSimCLR',
    data_preprocessor=None,
    backbone=dict(
        type='WideModel',
        in_channels=None, 
        stem_width=features_per_marker, 
        block_width=2, 
        layer_config=[1,1],
        late_fusion=False,
        drop_prob=0.05
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