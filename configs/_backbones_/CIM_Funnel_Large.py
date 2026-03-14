mix_channels = 512

backbone = dict(
    type='CIM_Funnel',
    in_channels=None,
    stem_width=16,
    block_width=4,
    sep_layer_config=[2, 2],     # 2 depthwise stages: 20→10→5px
    mix_n_blocks=8,              # 8 ConvNeXt-style MixBlocks in Phase 2
    mix_channels=mix_channels,   # 512ch — 2× wider than standard Funnel
    drop_prob=0.05,
    input_norm=False,
)
