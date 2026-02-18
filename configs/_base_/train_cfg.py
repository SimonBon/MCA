
n_linear = 200
n_cosine = 800
lr = 0.3

optimizer = dict(type='LARS', lr=lr, momentum=0.9, weight_decay=1e-5)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=n_linear+n_cosine
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=n_linear,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=n_cosine,
        eta_min=0.1 * lr,
        by_epoch=False,
        begin=n_linear,
        end=n_linear + n_cosine,
    )
]