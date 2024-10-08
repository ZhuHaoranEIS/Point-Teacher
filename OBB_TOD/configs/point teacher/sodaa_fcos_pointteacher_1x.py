_base_ = [
    '/home/zhr/SODA/configs/_base_/datasets/sodaarewrite.py', '/home/zhr/SODA/configs/_base_/schedules/schedule_1x.py',
    '/home/zhr/SODA/configs/_base_/default_runtime.py'
]

### Base
num_classes = 9
burn_in_step = 8000
ema_alpha = 0.999
angle_version = 'le90'
### MIL
num_stages = 1
mil_stack_conv = 0
top_k = 5
mil_neg_samples = 200
num_training_burninstep1 = 100
num_training_burninstep2 = 100
lamda = 1.0
_point_='center'
beta = 0.25
alpha = [0.01, 0.25] # reg, cls, aux
shape_list = [[20, 20, 0.5, 0.5], [10, 20, 0.5, 0.5], [10, 30, 0.5, 0.5],
              [40, 20, 0.5, 0.5], [30, 10, 0.5, 0.5], 
              [20, 50, 0.5, 0.5], [30, 20, 0.5, 0.5], [35, 40, 0.6, 0.5]]

detector = dict(
    type='RotatedFCOS_Student',
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
        num_outs=5, # P3 - P7
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    neck_agg=dict(
        type='PSAGG',
        num_aggregation=5, # epuals to num_outs
        in_channels=256,
        out_channels=256),
    bbox_head=dict(
        type='TS_P2RBRotatedFCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        mil_stack_conv=mil_stack_conv,
        feat_channels=256,
        strides=[8],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        top_k=top_k,
        beta=beta,
        angle_version=angle_version,
        num_stages=num_stages,
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[8]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox_burn1=dict(type='DIoULoss', loss_weight=1.0),
        # loss_bbox_burn2=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.1),
        loss_bbox_burn1=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_bbox_burn2=dict(type='DN_IoULoss', loss_weight=1.0, hyper=0.1),
        loss_bbox_denosing=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.2),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),),
    )

# model settings
model = dict(
    type='RotatedFCOS_TS',
    angle_version=angle_version,
    _model_=detector,
    ema_alpha=0.999,
    num_stages=num_stages,
    burn_in_step=burn_in_step,
    filter_score=0.0,
    lamda=lamda,
    _point_=_point_,
    alpha=alpha,
    shape_list=shape_list,
    num_training_burninstep1=num_training_burninstep1,
    num_training_burninstep2=num_training_burninstep2,
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='TopkAssigner',
            num_pre=1,
            topk=1,
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='PointCost', mode='L1', weight=1.0)),
        pseudo_assigner=dict(
            type='TopkAssigner',
            num_pre=3,
            topk=3,
            cls_cost=dict(type='FocalLossCost', weight=0.0),
            reg_cost=dict(type='PointCost', mode='L1', weight=1.0)),
        syn_assigner=dict(
            type='TopkAssigner',
            num_pre=3,
            topk=3,
            cls_cost=dict(type='FocalLossCost', weight=0.0),
            reg_cost=dict(type='PointCost', mode='L1', weight=1.0)),
        fuse_assigner=dict(
            type='FUSETopkAssigner',
            num_pre=5,
            topk=3,
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='PointCost', mode='L1', weight=1.0),
            location_cost=dict(type='InsiderCost', weight=1.0)),
        fine_proposal_cfg=[
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=None,
                 base_ratios=[1.0],
                 min_scale=0,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=mil_neg_samples),
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=None,
                 base_ratios=[1.0],
                 min_scale=0,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=mil_neg_samples),
        ],
        fine_proposal_extensive_cfg=[
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=None,
                 base_ratios=[1.0, 1.2, 1.3, 0.8, 0.6],
                 min_scale=4,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=0),
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=None,
                 base_ratios=[1.0, 1.3, 0.8],
                 min_scale=4,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=0),
        ],
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))


# img_norm_cfg = dict(
#     mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


optimizer = dict(lr=0.005)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

evaluation = dict(interval=12, metric='mAP')