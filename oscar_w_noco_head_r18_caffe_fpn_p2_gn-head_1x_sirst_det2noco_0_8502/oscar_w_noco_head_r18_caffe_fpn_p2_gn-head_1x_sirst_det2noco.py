out_inds = [0, 1]
sirst_version = 'sirstv2'
depth = 18
fpn_strides = [4, 8, 16, 32]
split_cfg = dict(
    train_split='splits/trainval_full.txt',
    val_split='splits/test_full.txt',
    test_split='splits/test_full.txt')
backbone_cfg = dict(
    stem_stride=2,
    max_pooling=True,
    strides=(1, 2, 2, 2),
    depths=(2, 2, 2, 2),
    block='basic',
    pretrained='open-mmlab://resnet18_v1c')
neck_cfg = dict(
    out_inds=[0, 1], in_channels=[64, 128, 256, 512], out_channels=256)
head_cfg = dict(in_channels=256)
dataset_type = 'SIRSTDet2NoCoDataset'
data_root = 'data/sirst/'
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='NoCoTargets', mode='det2noco'),
    dict(
        type='Normalize',
        mean=[111.89, 111.89, 111.89],
        std=[27.62, 27.62, 27.62],
        to_rgb=True),
    dict(type='OSCARPad', size_divisor=32),
    dict(type='OSCARFormatBundle'),
    dict(
        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_noco_map'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[111.89, 111.89, 111.89],
                std=[27.62, 27.62, 27.62],
                to_rgb=True),
            dict(type='OSCARPad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='SIRSTDet2NoCoDataset',
            ann_file=['data/sirst/splits/trainval_full.txt'],
            img_prefix=['data/sirst/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='NoCoTargets', mode='det2noco'),
                dict(
                    type='Normalize',
                    mean=[111.89, 111.89, 111.89],
                    std=[27.62, 27.62, 27.62],
                    to_rgb=True),
                dict(type='OSCARPad', size_divisor=32),
                dict(type='OSCARFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_noco_map'])
            ])),
    val=dict(
        type='SIRSTDet2NoCoDataset',
        ann_file='data/sirst/splits/test_full.txt',
        img_prefix='data/sirst/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[111.89, 111.89, 111.89],
                        std=[27.62, 27.62, 27.62],
                        to_rgb=True),
                    dict(type='OSCARPad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SIRSTDet2NoCoDataset',
        ann_file='data/sirst/splits/test_full.txt',
        img_prefix='data/sirst/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[111.89, 111.89, 111.89],
                        std=[27.62, 27.62, 27.62],
                        to_rgb=True),
                    dict(type='OSCARPad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='OSCARNet',
    backbone=dict(
        type='FlexResNet',
        depths=(2, 2, 2, 2),
        block='basic',
        stem_stride=2,
        max_pooling=True,
        in_channels=3,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FlexFPN',
        out_inds=[0, 1],
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    bbox_head=dict(
        type='OSCARNoCoHead',
        num_classes=1,
        stride_ratio=1.5,
        in_channels=256,
        regress_ranges=((-1, 100000000.0), ),
        stacked_convs=4,
        feat_channels=128,
        strides=[4, 8],
        loss_coarse_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_refine_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_refine_noco=dict(
            type='RegQualityFocalLoss',
            beta=2.0,
            use_sigmoid=True,
            loss_weight=1000.0),
        loss_coarse_bbox=dict(type='DIoULoss', loss_weight=1.0),
        loss_refine_bbox=dict(type='DIoULoss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
optimizer = dict(
    type='AdamW',
    lr=0.002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=6))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = 'work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco_gpu_0'
auto_resume = False
gpu_ids = [0]
