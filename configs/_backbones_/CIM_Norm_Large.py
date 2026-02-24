features_per_marker = 64

backbone = dict(
    type='WideModel',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[2, 2],
    drop_prob=0.05,
    input_norm=True,
)
