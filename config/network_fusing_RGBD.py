checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet_rgb_depth_attention_cbam_d_b',
        depth=53,
        out_indices=(3, 4, 5),
        pretrained=None,
        init_cfg=None),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = '../data/ht_cumt_rgbd/'
img_norm_cfg = dict(
    mean=[0, 0, 0, 0], std=[255.0, 255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_rgb_depth'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=[0, 0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion_rgb_depth'),
    dict(
        type='Normalize',
        mean=[0, 0, 0, 0],
        std=[255.0, 255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile_rgb_depth'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0, 0],
                std=[255.0, 255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('person', )
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file='../data/ht_cumt_rgbd/annotations/instances_train2014.json',
        img_prefix='../data/ht_cumt_rgbd/train2014/',
        pipeline=[
            dict(type='LoadImageFromFile_rgb_depth'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=[0, 0, 0, 0],
                to_rgb=True,
                ratio_range=(1, 2)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(320, 320), (416, 416)],
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion_rgb_depth'),
            dict(
                type='Normalize',
                mean=[0, 0, 0, 0],
                std=[255.0, 255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('person', ),
        img_prefix_depth='../data/ht_cumt_rgbd/depth_train/'),
    val=dict(
        type='CocoDataset',
        ann_file='../data/ht_cumt_rgbd/annotations/instances_val2014.json',
        img_prefix='../data/ht_cumt_rgbd/val2014/',
        pipeline=[
            dict(type='LoadImageFromFile_rgb_depth'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0, 0],
                        std=[255.0, 255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('person', ),
        img_prefix_depth='../data/ht_cumt_rgbd/depth_val/'),
    test=dict(
        type='CocoDataset',
        ann_file='../data/ht_cumt_rgbd/annotations/instances_val2014.json',
        img_prefix='../data/ht_cumt_rgbd/val2014/',
        pipeline=[
            dict(type='LoadImageFromFile_rgb_depth'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0, 0],
                        std=[255.0, 255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('person', ),
        img_prefix_depth='../data/ht_cumt_rgbd/depth_val/'))
optimizer = dict(type='SGD', lr=0.00075, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[218, 246])
runner = dict(type='EpochBasedRunner', max_epochs=273)
evaluation = dict(interval=1, metric=['bbox'])
work_dir = '../work_dirs_rgb_depth_attention_cbam_d_b'
auto_resume = False
gpu_ids = [0]
