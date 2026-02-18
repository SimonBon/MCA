train_augmentation = [
    dict(type='C_RandomFlip', prob=0.5, horizontal=True, vertical=True),
    dict(type='C_RandomAffine', angle=(0, 360), scale=(0.8, 1.2), shift=(0, 0), order=1),
    dict(type='C_RandomIntensity', low=2/3, high=3/2, clip=True),
    dict(type='C_RandomNoise', mean=(0, 0), std=(0, 0.03), clip=True),
    dict(type='C_RandomChannelMixup', mixup_prob=0.1),
    dict(type='C_RandomChannelDrop', drop_prob=0.1),
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[None, None]),
    dict(type='C_PackInputs'),
]


