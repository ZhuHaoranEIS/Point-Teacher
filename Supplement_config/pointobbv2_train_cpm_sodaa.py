_base_ = [
    '../_base_/datasets/sodaa.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

data_root = '/data/zhr/SODA/SODA-A/'

store_dir = '/home/zhr/PointOBB-v2/configs/pointobbv2/cpm'

classes = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
           'ship', 'container', 'storage-tank', 'swimming-pool',
           'windmill')

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1200, 1200)),
    dict(type='RRandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'divData/train/Annotations_filter/',
        img_prefix=data_root + 'divData/train/Images_filter/',
        classes=classes,
        version=angle_version),
    val=dict(
        ann_file=data_root + 'divData/train/Annotations_filter/',
        img_prefix=data_root + 'divData/train/Images_filter/',
        classes=classes,
        version=angle_version),
    test=dict(
        ann_file=data_root + 'divData/val/Annotations_filter/',
        img_prefix=data_root + 'divData/val/Images_filter/',
        classes=classes,
        version=angle_version,
        samples_per_gpu=4))

# model settings
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CPMHead',
        num_classes=9,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, 1e8)),
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        visualize=True,
        store_dir=store_dir,
        cls_weight=1.0,
        thresh1=6,
        alpha=1.5
        ),
    test_cfg=dict(
        store_dir=store_dir,
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
find_unused_parameters = True
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=6)
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4])
evaluation = dict(interval=6, metric='mAP')
optimizer = dict(lr=0.05)