import os

n_linear = 5   if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes') else 200
n_cosine = 5   if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes') else 800
lr = 3e-4

optimizer = dict(type='AdamW', lr=lr, weight_decay=0.05)

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
        eta_min=1e-6,
        by_epoch=False,
        begin=n_linear,
        end=n_linear + n_cosine,
    )
]
