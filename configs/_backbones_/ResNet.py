features_per_marker = 4   # base_width=64, output=64*4=256 = 41 * ~6, closest match

backbone = dict(
    type='ResNetBaseline',
    in_channels=None,
    base_width=64,
    drop_prob=0.05,
)
