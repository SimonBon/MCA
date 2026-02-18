model = dict(
    type='MVSimCLR',
    data_preprocessor=None,
    backbone=None,
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
    ),
)
