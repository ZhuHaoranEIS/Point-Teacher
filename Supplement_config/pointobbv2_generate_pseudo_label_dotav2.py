_base_ = [
    '../_base_/datasets/dotav2.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

data_root = '/data/xc/DOTA-v2/'

store_dir = '/home/zhr/PointOBB-v2/configs/pointobbv2/cpm_dotav2'

classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 
           'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 
           'helicopter', 'container-crane', 'airport', 'helipad')

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'train_obb/split_images/annfiles/',
        img_prefix=data_root + 'train_obb/split_images/images/',
        classes=classes,
        version=angle_version),
    val=dict(
        ann_file=data_root + 'val_obb/split_images/annfiles/',
        img_prefix=data_root + 'val_obb/split_images/images/',
        classes=classes,
        version=angle_version),
    test=dict(
        ann_file=data_root + 'val_obb/split_images/annfiles/',
        img_prefix=data_root + 'val_obb/split_images/images/',
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
    # store_dir='rotated_fcos_r50_fpn_1x_dota_le90_2',
    bbox_head=dict(
        type='PseudoLabelHead',
        num_classes=18,
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

    # classes = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
    #        'ship', 'container', 'storage-tank', 'swimming-pool',
    #        'windmill')
    train_cfg=dict(
        store_dir=store_dir,
        cls_weight=1.0,
        # thresh3=[0.03, 0.04, 0.1, 0.01, 0.10, 0.06, 0.08, 0.02, 0.01, 0.03, 0.005, 0.02, 0.05, 0.1, 0.015],
        # thresh3=[0.05, 0.04, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04],
        thresh3=[0.04, # Plane
                 0.04, # Baseball-diamond
                 0.05, # Bridge
                 0.01, # Ground-track-field
                 0.2, # Small-vehicle
                  
                 0.08, # Large-vehicle
                 0.08, # Ship
                 0.03, # Tennis-court
                 0.01, # Basketball-court
                 0.02, # Storage-tank
                  
                 0.01, # Soccer-ball-field
                 0.02, # Roundabout
                 0.1, # Harbor
                 0.1, # Swimming-pool
                 0.01, # Helicopter
                 
                 0.01, # Container-crane
                 0.005, # Airport
                 0.02, # Helipad
        ],
        pca_length=40,
        store_ann_dir='/data/zhr/DOTAv2/obb/pointobbv2/dotav2_cpm_p3/',
        multiple_factor=1/4
        ),
    test_cfg=dict(
        store_dir=store_dir,
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=3000))
find_unused_parameters = True

runner = dict(type='EpochBasedRunner', max_epochs=7)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4])

evaluation = dict(interval=3, metric='mAP')
optimizer = dict(lr=0.0)