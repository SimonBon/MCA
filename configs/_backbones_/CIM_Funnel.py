mix_channels = 256

backbone = dict(
    type='CIM_Funnel',
    in_channels=None,       # set per-dataset in experiment config
    stem_width=16,          # per-marker feature expansion in Phase 1
    block_width=4,          # FFN expansion ratio inside ConvBlocks
    sep_layer_config=[2, 2],     # 2 depthwise stages: 20→10→5px (MIBI_TNBC cutter_size=20)
    mix_n_blocks=2,         # cross-channel ConvBlocks in Phase 2
    mix_channels=mix_channels,   # project from C*stem_width → 256 at transition
    drop_prob=0.05,
    input_norm=False,
)
