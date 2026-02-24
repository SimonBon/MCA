features_per_marker = 32

backbone = dict(
    type='WideModelProgressiveFusion',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[1, 1],
    drop_prob=0.05,
    input_norm=True,
)
