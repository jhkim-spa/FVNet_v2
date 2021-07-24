model = dict(
    type='FVNet',
    fv_backbone=dict(
        type='UNet',
        in_channels=5,
        num_outs=1),
    bbox_head=dict(
        type='FVNetHead',
        num_classes=1,
        feat_channels=67,
        bbox_coder=dict(
            type='FVNetBBoxCoder',
            prior_size=[1.6, 3.9, 1.56],
            code_size=8),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
    train_cfg=dict(
        assigner=dict(type='InBoxAssigner'),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.3,
        min_bbox_size=0,
        max_num=50)))

find_unused_parameters = True
