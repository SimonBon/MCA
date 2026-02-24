train_aug_mid = [
    dict(type='C_RandomFlip', prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine', angle=(0, 360), scale=(0.75, 1.33), shift=(-0.05, 0.05), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.5, 2.0), shift=(-0.1, 0.1), clip=True),
    dict(type='C_RandomBackgroundGradient', strength=(-0.1, 0.1), clip=True),
    dict(type='C_RandomNoise', mean=(0, 0), std=(0, 0.035), clip=True),
    dict(type='C_RandomChannelCopy', copy_prob=0.03),
    dict(type='C_RandomChannelMixup', mixup_prob=0.03),
    dict(type='C_RandomChannelDrop', drop_prob=0.06),
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[None, None]),
    dict(type='C_PackInputs'),
]
