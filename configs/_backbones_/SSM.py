features_per_marker = 32

backbone = dict(
    type='SharedStemModel',
    in_channels=None,
    stem_width=32,
    block_width=2,
    n_layers=2,
    late_fusion=False,
)
