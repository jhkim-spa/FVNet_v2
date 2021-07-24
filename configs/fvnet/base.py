_base_ = [
    './_base_/models/fvnet.py',
    './_base_/datasets/fv_kitti-3d-car_620x190.py',
    '../_base_/schedules/seg_cosine_50e.py',
    '../_base_/default_runtime.py'
]

log_config = dict(interval=10)