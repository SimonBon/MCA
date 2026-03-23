mix_channels = 512

backbone = dict(
    type='CIM_Funnel',
    in_channels=None,
    stem_width=16,
    block_width=4,
    sep_layer_config=[2, 2],
    mix_n_blocks=8,
    mix_channels=mix_channels,
    drop_prob=0.05,
    input_norm=False,
    relative_norm=True,
)
