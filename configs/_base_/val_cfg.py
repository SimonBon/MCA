val_augmentation = [
    dict(type='C_CentralCutter', size=None),
    dict(type='C_ToTensor')
]

val_pipeline = [
    dict(type='C_MultiView', n_views=[1], transforms=[None]),
    dict(type='C_PackInputs'),
]

custom_hooks = [dict(
    type='EvaluateModel', 
    train_indicies=None,
    val_indicies=None,
    dataset_kwargs=None,
    short=False)
]