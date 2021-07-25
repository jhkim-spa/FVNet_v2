# optimizer
lr = 0.001
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.001)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    gamma=0.5,
    step=[50, 75, 90, 100])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
