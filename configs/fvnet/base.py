_base_ = [
    './_base_/models/fvnet.py',
    './_base_/datasets/fv_kitti-3d-car.py',
    './_base_/schedules/cosine_200e.py',
    '../_base_/default_runtime.py'
]

log_config = dict(interval=10)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2)