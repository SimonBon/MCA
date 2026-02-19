model = dict(
    type='MVBYOL',
    momentum=0.996,
    pred_in_channels=256,   # must match neck out_channels
    pred_hid_channels=512,
    pred_out_channels=256,
    data_preprocessor=None,
    backbone=None,
    neck=dict(
        type='NonLinearNeck',
        in_channels=None,
        hid_channels=512,
        out_channels=256,
        num_layers=2,
        with_avg_pool=False,
    ),
)
