train_aug_strong = [
    dict(type='C_RandomFlip', prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine', angle=(0, 360), scale=(0.66, 1.5), shift=(-0.1, 0.1), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.33, 3), shift=(-0.15, 0.15), clip=True),
    dict(type='C_RandomBackgroundGradient', strength=(-0.15, 0.15), clip=True),
    dict(type='C_RandomNoise', mean=(0, 0), std=(0, 0.05), clip=True),
    dict(type='C_RandomChannelCopy', copy_prob=0.05),
    dict(type='C_RandomChannelMixup', mixup_prob=0.05),
    dict(type='C_RandomChannelDrop', drop_prob=0.2),
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_aug_weak = [
    dict(type='C_RandomFlip', prob=0.3, horizontal=True, vertical=True),
    dict(type='C_RandomAffine', angle=(0, 360), scale=(0.9, 1.1), shift=(0, 0), order=1),
    dict(type='C_RandomChannelShiftScale', scale=(0.9, 1.1), shift=(-0.05, 0.05), clip=True),
    dict(type='C_RandomBackgroundGradient', strength=(0.0, 0.05), clip=True),
    dict(type='C_RandomNoise', mean=(0, 0), std=(0, 0.02), clip=True),
    dict(type='C_RandomChannelCopy', copy_prob=0.01),
    dict(type='C_RandomChannelMixup', mixup_prob=0.01),
    dict(type='C_RandomChannelDrop', drop_prob=0.05),
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[None, None]),
    dict(type='C_PackInputs'),
]
