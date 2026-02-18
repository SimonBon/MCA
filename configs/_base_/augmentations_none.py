train_aug_strong = [
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_aug_weak = [
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

train_pipeline = [
    dict(type='C_MultiView', n_views=[1, 1], transforms=[None, None]),
    dict(type='C_PackInputs'),
]
