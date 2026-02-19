features_per_marker = 32

backbone = dict(
    type='WideModel',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[2, 2],
    late_fusion=True,
    drop_prob=0.05,
)
