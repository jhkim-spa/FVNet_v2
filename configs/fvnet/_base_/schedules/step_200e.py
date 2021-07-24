# optimizer
lr = 0.003
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup=None,
    step=[134, 183])
momentum_config = None
# runtime settings
total_epochs = 200
