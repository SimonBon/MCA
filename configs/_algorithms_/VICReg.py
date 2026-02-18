model = dict(
    type='MVVICReg',
    data_preprocessor=None,
    sim_coeff=25.0,
    std_coeff=25.0,
    cov_coeff=1.0,
    gamma=1.0,
    backbone=None,
    neck=dict(
        type='NonLinearNeck',
        in_channels=None,
        hid_channels=512,
        out_channels=512,
        num_layers=2,
        with_avg_pool=False,
    ),
)
