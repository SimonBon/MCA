# stem_width=5 gives ~1.17M params, matching WideModel (stem_width=32, ~1.11M)
# Early fusion standard convs have O(C^2) params vs depthwise O(C), so stem_width
# must be much smaller to achieve a fair parameter-matched comparison.
features_per_marker = 5

backbone = dict(
    type='EarlyFusionModel',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[1, 1],
    late_fusion=False,
    drop_prob=0.05,
)
