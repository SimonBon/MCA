# stem_width=5 gives ~1.17M params, matching WideModel (stem_width=32, ~1.11M)
features_per_marker = 5

backbone = dict(
    type='ProjectionFusionModel',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[1, 1],
    drop_prob=0.05,
)
