_base_ = [
    '/home/zhr/mmdet-rfla/configs/_base_/datasets/aitodv2_detection_point.py',
    '/home/zhr/mmdet-rfla/configs/_base_/schedules/schedule_1x.py', 
    '/home/zhr/mmdet-rfla/configs/_base_/default_runtime.py'
]

### Base
num_classes = 8
burn_in_step = 4000
ema_alpha = 0.999
### MIL
num_stages = 1
mil_stack_conv = 0
top_k = 1
mil_neg_samples = 200
num_training_burninstep1 = 75
num_training_burninstep2 = 75
lamda = 0.5
_point_ = 1.0
beta = 0.25
alpha = [0.01, 0.25] # reg, cls, aux
shape_list = [[20, 20, 0.5, 0.5], [10, 20, 0.5, 0.5], [30, 80, 0.5, 0.5],
              [20, 50, 0.5, 0.5], [30, 120, 0.5, 0.5], [30, 40, 0.5, 0.5]]

# model settings
detector = dict(
    type='Student_FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    neck_agg=dict(
        type='PSAGG',
        num_aggregation=5, # epuals to num_outs
        in_channels=256,
        out_channels=256),
    bbox_head=dict(
        type='TS_P2BFCOSHead',
        norm_cfg=None,
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        mil_stack_conv=mil_stack_conv,
        feat_channels=256,
        strides=[8],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        beta=beta,
        top_k=top_k,
        num_stages=num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[8]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_burn1=dict(type='DIoULoss', loss_weight=1.0),
        loss_bbox_burn2=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.1),
        loss_bbox_denosing=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

model = dict(
    type='TS_P2B_FCOS',
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
                 base_ratios=[1.0, 1.3, 0.8],
                 min_scale=0,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=200),
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=None,
                 base_ratios=[1.0],
                 min_scale=4,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=mil_neg_samples),
        ],
        fine_proposal_extensive_cfg=[
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=[0.1],
                 base_ratios=[1.0, 1.3, 0.7],
                 min_scale=4,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=0),
            dict(gen_mode='refine',
                 gen_proposal_mode='fix_gen',
                 cut_mode=None,
                 shake_ratio=[0.1],
                 base_ratios=[1.0, 1.2, 1.3, 0.8, 0.7],
                 min_scale=16,
                 pos_iou_thr=0.3,
                 neg_iou_thr=0.3,
                 gen_num_neg=0),
        ],
    ),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))


img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01/2, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
