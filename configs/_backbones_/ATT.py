features_per_marker = 32

backbone = dict(
    type='MCIANet',
    in_channels=None,
    input_size=None,
    stem_dim=features_per_marker,
    n_heads=4,
    stem_blocks=2,
    stage1_blocks=2,
    stage2_blocks=2,
    expansion=2,
    drop_prob=0.05,
    sigma_fraction=0.35,
    spatial_init_mode='ones',
)
