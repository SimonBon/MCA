# WideModelLateFusion: same sep_stages as CIM, then a 2-layer MLP mixer.
# out_dim matches CIM (in_channels * stem_width) so the same neck config applies.
features_per_marker = 32

backbone = dict(
    type='WideModelLateFusion',
    in_channels=None,
    stem_width=features_per_marker,
    block_width=2,
    layer_config=[1, 1],
    mlp_ratio=4,
    out_channels=None,   # defaults to in_channels * stem_width — same as CIM
    drop_prob=0.05,
    input_norm=False,
)
