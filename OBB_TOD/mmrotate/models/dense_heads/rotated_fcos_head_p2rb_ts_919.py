# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, distance2bbox, bbox2distance, 
                        build_assigner, build_sampler, bbox_overlaps)

from mmdet.core.bbox.match_costs import build_match_cost
from ...core.bbox.transforms import obb2poly, poly2obb

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead

from ...core.bbox.iou_calculators import build_iou_calculator, rbbox_overlaps

import numpy as np
import itertools

from mmdet.models.utils import build_linear_layer
import os
from ..detectors.data_augument_bank import imshow_det_rbboxes
import cv2

import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from ..builder import build_roi_extractor
from mmrotate.core import rbbox2roi

INF = 1e8


@ROTATED_HEADS.register_module()
class TS_P2RBRotatedFCOSHead(RotatedAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        scale_angle (bool): If true, add scale to angle pred branch. Default: True.
        h_bbox_coder (dict): Config of horzional bbox coder, only used when separate_angle is True.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_angle (dict): Config of angle loss, only used when separate_angle is True.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, INF),),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 separate_angle=False,
                 scale_angle=True,
                 h_bbox_coder=dict(type='DistancePointBBoxCoder'),
                 mil_stack_conv=1,
                 top_k=3,
                 bbox_roi_extractor=dict(
                     type='RotatedSingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlignRotated',
                         out_size=7,
                         sample_num=2,
                         clockwise=True),
                     out_channels=256,
                     featmap_strides=[8]),
                 loss_mil1=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25*2,
                     loss_type='binary_cross_entropy'),
                 loss_mil2=dict(
                         type='MILLoss',
                         binary_ins=False,
                         loss_weight=0.25*2,
                         loss_type='gfocal_loss'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_angle=dict(type='L1Loss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=dict(
                     assigner=dict(
                         type='TopkAssigner',
                         topk=3,
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='PointCost', mode='L1', weight=5.0)),
                     pseudo_assigner=dict(
                         type='TopkAssigner',
                         num_pre=5,
                         topk=3,
                         cls_cost=dict(type='FocalLossCost', weight=1.0),
                         reg_cost=dict(type='PointCost', mode='L1', weight=1.0)),
                     syn_assigner=dict(
                         type='TopkAssigner',
                         num_pre=5,
                         topk=3,
                         cls_cost=dict(type='FocalLossCost', weight=1.0),
                         reg_cost=dict(type='PointCost', mode='L1', weight=1.0)),
                     fuse_assigner=dict(
                         type='PHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=1.),
                         center_cost=dict(type='CrossEntropyLossCost', weight=1., use_sigmoid=True),
                         location_cost=dict(type='InsiderCost', weight=1.0))),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.separate_angle = separate_angle
        self.is_scale_angle = scale_angle
        self.mil_stack_conv = mil_stack_conv
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        # Angle predict length
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)   
            fuse_assigner = train_cfg['fuse_assigner']
            self.fuse_assigner = build_assigner(fuse_assigner) 
            syn_assigner = train_cfg['syn_assigner']
            self.syn_assigner = build_assigner(syn_assigner)
            pseudo_assigner = train_cfg['pseudo_assigner']
            self.pseudo_assigner = build_assigner(pseudo_assigner)
            self.preds_decoder = build_bbox_coder(dict(type='DistanceAnglePointCoder'))
        
        self.loss_angle = build_loss(loss_angle)
        self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
        self.loss_hbbox = build_loss(dict(type='DIoULoss', loss_weight=1.0))

        ### mil head
        self.loss_mil1 = build_loss(loss_mil1)
        self.loss_mil2 = build_loss(loss_mil2)
        self.num_stages = 2
        self.merge_mode = 'weighted_clsins'
        self.topk1 = top_k  # 3
        self.topk2 = top_k  # 3
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_mil = nn.ModuleList()
        for i in range(self.mil_stack_conv):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.conv_mil.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

        ### mil head
        self.num_shared_fcs = 2
        self.with_avg_pool = False
        self.fc_out_channels = 1024
        self.conv_out_channels = self.feat_channels
        self.roi_feat_area = 7 * 7

        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim
        # add cls specific branch
        _, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(0, 0, self.shared_out_channels)
        # add ins specific branch
        _, self.ins_fcs, self.ins_last_dim = \
            self._add_conv_fc_branch(0, 0, self.shared_out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.ModuleList()
        self.fc_ins = nn.ModuleList()
        num_cls = self.num_classes
        self.fc_cls = build_linear_layer(
                    dict(type='Linear'),
                    in_features=self.cls_last_dim,
                    out_features=num_cls)
        self.fc_ins = build_linear_layer(
                    dict(type='Linear'),
                    in_features=self.ins_last_dim,
                    out_features=num_cls)
    
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                angle_preds (list[Tensor]): Box angle for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=feats[0].dtype,
            device=feats[0].device)
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, all_level_points)

    def forward_single(self, x, scale, stride, points):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        B, _, H, W = x.shape
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = torch.clamp(bbox_pred, min=0)
            bbox_pred = bbox_pred * stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()

        return cls_score, bbox_pred, angle_pred, centerness, points
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses',
                  'all_level_points'))
    def get_pseudo_bbox(self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            all_level_points,
            gt_points,
            gt_labels,
            gt_bboxes,
            filter_scores,
            img_metas,
            img_list,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores, flatten_bbox_preds, flatten_centernesses = self.concat_per_img(num_imgs, cls_scores, bbox_preds, angle_preds, centernesses)

        pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds = self.gnerate_pseudo(all_level_points, gt_points, gt_labels, gt_bboxes, filter_scores, img_metas, img_list, 
                                                                                            flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
    
        return pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses',
                  'all_level_points'))
    def loss_pseudo(self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            all_level_points,
            gt_points,
            gt_labels,
            pseudo_points, 
            pseudo_labels,
            pseudo_bboxes,
            gt_augument_ignore,
            img_metas,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores, flatten_bbox_preds, flatten_centernesses = self.concat_per_img(num_imgs, cls_scores, bbox_preds, angle_preds, centernesses)

        labels_reg, bbox_targets, angle_targets, labels, weights = self.get_target_pseudo(all_level_points, gt_points, gt_labels, pseudo_bboxes, img_metas, gt_augument_ignore,
                                                                                 pseudo_points, pseudo_labels, 
                                                                                 flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_weights = torch.cat(weights)
        flatten_labels_reg = torch.cat(labels_reg)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, weight=flatten_weights, avg_factor=num_pos)
        
        pos_inds = ((flatten_labels_reg >= 0)
                    & (flatten_labels_reg < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.bbox_coder
            pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                        dim=-1)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_cls, loss_bbox, loss_centerness
    
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses',
                  'all_level_points'))
    def loss(self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            all_level_points,
            gt_bboxes,
            img_metas,
            img_syn_list,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores, flatten_bbox_preds, flatten_centernesses = self.concat_per_img(num_imgs, cls_scores, bbox_preds, angle_preds, centernesses)

        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, img_syn_list, img_metas, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
        
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.bbox_coder
            pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                        dim=-1)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_bbox, loss_centerness
    
    def get_targets(self, points, gt_bboxes_list, img_syn_list, img_metas, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        concat_cls_scores, concat_bbox_preds, concat_centernesses = self.concat_prediction(num_levels, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            img_syn_list,
            img_metas,
            concat_cls_scores,
            concat_bbox_preds,
            concat_centernesses,
            points=concat_points,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)
    
    def _get_target_single(self, gt_bboxes, img_syn, img_metas, cls_scores, bbox_preds, centerness, points,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_labels = torch.zeros(num_gts).to(gt_bboxes.device).long()
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        assign_result = self.syn_assigner.assign(points, cls_scores, 
                                            gt_bboxes, gt_labels, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        labels = self.num_classes * torch.ones(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]

        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inds = inds * 0 
        inds[pos_inds] = assign_result.gt_inds[pos_inds] - 1
    
        bbox_targets = bbox_targets[range(num_points), inds]
        angle_targets = gt_angle[range(num_points), inds]

        return labels, bbox_targets, angle_targets
    
    def get_target_pseudo(self, points, gt_points_list, gt_labels_list, pseudo_bboxes, img_metas, gt_augument_ignore, 
                          pseudo_points, pseudo_labels,
                          flatten_cls_scores, flatten_bbox_preds, flatten_centernesses):
        num_levels = len(points)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        concat_points = torch.cat(points, dim=0)
        concat_cls_scores, concat_bbox_preds, concat_centernesses = self.concat_prediction(num_levels, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)

        results = multi_apply(
                self._get_target_pseudo_single,
                gt_points_list,
                gt_labels_list,
                pseudo_points,
                pseudo_labels,
                pseudo_bboxes,
                concat_cls_scores,
                concat_bbox_preds,
                concat_centernesses,
                img_metas,
                gt_augument_ignore,
                points=concat_points,
                num_points_per_lvl=num_points)
        
        labels_reg_list, bbox_targets_list, angle_targets_list, labels_list, weights_list = results

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        weights_list = [weights.split(num_points, 0) for weights in weights_list]
        labels_reg_list = [labels.split(num_points, 0) for labels in labels_reg_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_weights = []
        concat_lvl_labels_reg = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_weights.append(
                torch.cat([weights[i] for weights in weights_list]))
            concat_lvl_labels_reg.append(
                torch.cat([labels[i] for labels in labels_reg_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels_reg, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_labels, concat_lvl_weights)
               
               
    def _get_target_pseudo_single(self, gt_points, gt_labels, 
                                  pseudo_points, pseudo_labels, pseudo_bboxes, 
                                  cls_scores, bbox_preds, centernesses, 
                                  img_metas, gt_augument_ignore, points, num_points_per_lvl):
        num_points = points.size(0)
        imagefile = img_metas['ori_filename']
        ### clssification
        assign_result = self.assigner.assign(points, cls_scores, 
                                            gt_points, gt_labels, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        labels = self.num_classes * torch.ones(num_points, dtype=torch.long, device=pseudo_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]

        ### weight
        weights = torch.ones_like(labels).float()

        ### regression
        num_gts = len(pseudo_labels)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   pseudo_bboxes.new_zeros((num_points, 4)), \
                   pseudo_bboxes.new_zeros((num_points, 1)), \
                   labels, \
                   weights
        
        assign_result = self.pseudo_assigner.assign(points, cls_scores, 
                                            pseudo_bboxes, pseudo_labels, gt_bboxes_ignore=None)

        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        
        labels_reg = self.num_classes * torch.ones(num_points, dtype=torch.long, device=pseudo_bboxes.device)
        labels_reg[pos_inds] = assign_result.labels[pos_inds]

        points = points[:, None, :].expand(num_points, num_gts, 2)
        pseudo_bboxes = pseudo_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(pseudo_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inds = inds * 0 
        inds[pos_inds] = assign_result.gt_inds[pos_inds] - 1
    
        bbox_targets = bbox_targets[range(num_points), inds]
        angle_targets = gt_angle[range(num_points), inds]

        return labels_reg, bbox_targets, angle_targets, labels, weights
    

    def gnerate_pseudo(self, points, gt_points_list, gt_labels_list, gt_bboxes_list, filter_scores, img_metas, img_list, 
                    flatten_cls_scores, flatten_bbox_preds, flatten_centernesses):
        num_levels = len(points)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        concat_points = torch.cat(points, dim=0)
        concat_cls_scores, concat_bbox_preds, concat_centernesses = self.concat_prediction(num_levels, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)

        results = multi_apply(
                self._gnerate_pseudo_single,
                gt_points_list,
                gt_labels_list,
                gt_bboxes_list,
                concat_cls_scores,
                concat_bbox_preds,
                concat_centernesses,
                img_metas,
                img_list,
                filter_scores=filter_scores,
                points=concat_points,
                num_points_per_lvl=num_points)
        
        pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds = results
        
        return pseudo_bboxes, pseudo_points, pseudo_labels, sum(mean_ious_pred)/len(mean_ious_pred), valid_inds
               
               
    def _gnerate_pseudo_single(self, gt_points, gt_labels, gt_bboxes, cls_scores, bbox_preds, centernesses, 
                            img_metas, img_list, filter_scores, points, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = len(gt_labels)
        if num_gts == 0:
            return torch.empty((0, 5), device=gt_labels.device), \
                   torch.empty((0, 2), device=gt_labels.device), \
                   torch.empty((0, 1), device=gt_labels.device), \
                   0.0, \
                   None
        
        cls_scores_activate = cls_scores.clone().detach().sigmoid()
        bbox_preds = self.bbox_coder.decode(points, bbox_preds)
        assign_result = self.fuse_assigner.assign(bbox_preds, points, cls_scores, centernesses,
                                            gt_points, gt_labels, gt_bboxes_ignore=None)
        
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        
        _, sorted_indices = torch.sort(inds[pos_inds]-1)
        pos_inds = pos_inds[sorted_indices]
        
        labels = torch.zeros(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]
        
        cls_scores_activate = cls_scores_activate[range(num_points), labels].reshape(-1,1)
        
        A = bbox_preds[pos_inds,:]
        B = assign_result.gt_inds[pos_inds] - 1
        C = cls_scores_activate[pos_inds,:].reshape(-1)
        
        assign_nums = torch.bincount(B, minlength=num_gts)
        pseudo_bboxes = 8 * torch.ones_like(gt_bboxes)
        pseudo_scores = torch.zeros(num_gts).to(cls_scores_activate.device).to(cls_scores_activate.dtype)
        pseudo_points = gt_points.clone()
        pseudo_bboxes[:,:2] = pseudo_points
        pseudo_bboxes[:,-1] *= 0
        
        B_one_hot = torch.nn.functional.one_hot(B, num_classes=num_gts) # M * num_gts
        
        pseudo_bboxes_sum = torch.matmul(B_one_hot.T.float(), A * C.unsqueeze(1))  # num_gts * 5
        pseudo_scores_sum = torch.matmul(B_one_hot.T.float(), C)   # num_gts * 1
        
        zero_inds = (assign_nums != 0).nonzero().reshape(-1)
        
        pseudo_bboxes[zero_inds,:] = pseudo_bboxes_sum[zero_inds,:] / pseudo_scores_sum[zero_inds].unsqueeze(1)
        pseudo_scores[zero_inds] = pseudo_scores_sum[zero_inds] / assign_nums[zero_inds]
        pseudo_points[zero_inds,:] = pseudo_bboxes[zero_inds,:2]
        
        mean_ious_pred = rbbox_overlaps(pseudo_bboxes[zero_inds,:], gt_bboxes[zero_inds,:], mode='iou', is_aligned=True).mean()
        
        filter_inds = (pseudo_scores >= filter_scores).nonzero().reshape(-1)
        
        valid_inds = torch.Tensor(list(set(zero_inds.tolist()) & set(filter_inds.tolist()))).long().to(zero_inds.device)

        pseudo_labels = gt_labels
        
        return pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
                   all_level_points,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=torch.ones_like(mlvl_centerness))
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centerness'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0].clamp(min=0.01) / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0].clamp(min=0.01) / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
    

    def concat_per_img(self, num_imgs, cls_scores, bbox_preds, angle_preds, centernesses):
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_hbbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centernesses = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for centerness in centernesses
        ]
        flatten_bbox_preds = [
            torch.cat([flatten_hbbox_preds[i],flatten_angle_preds[i]], dim=-1)
            for i in range(len(flatten_angle_preds))
        ]
        return flatten_cls_scores, flatten_bbox_preds, flatten_centernesses
    
    def concat_prediction(self, num_levels, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses):
        concat_cls_scores = []
        concat_bbox_preds = []
        concat_centernesses = []
        for i in range(flatten_cls_scores[0].shape[0]):
            temp_scores = []
            temp_bbox_preds = []
            temp_centernesses = []
            for j in range(num_levels):
                temp_scores.append(flatten_cls_scores[j][i,:,:])
                temp_bbox_preds.append(flatten_bbox_preds[j][i,:,:])
                temp_centernesses.append(flatten_centernesses[j][i,:,:])
            concat_cls_scores.append(torch.cat(temp_scores, dim=0))
            concat_bbox_preds.append(torch.cat(temp_bbox_preds, dim=0))
            concat_centernesses.append(torch.cat(temp_centernesses, dim=0))

        return concat_cls_scores, concat_bbox_preds, concat_centernesses
    
    def forward_mil(self, feats):
        results = multi_apply(self.forward_mil_single, feats)
        return results[0]

    def forward_mil_single(self, x):
        mil_feat = x
        if len(self.conv_mil) == 0:
            return mil_feat, x
        for mil_layer in self.conv_mil:
            mil_feat = mil_layer(mil_feat)
        return mil_feat, x
    
    def forward_mil_head(self, x, stage):
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        x_cls = x
        x_ins = x
        cls_score = self.fc_cls(x_cls)
        ins_score = self.fc_ins(x_ins)

        return cls_score, ins_score, None
    
    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_mil(self,
                 stage,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 reg_box,
                 labels,
                 gt_boxes,
                 label_weights,
                 retrain_weights,
                 reduction_override=None):
        losses = dict()
        if cls_score.numel() > 0:
            label_valid = proposals_valid_list
            cls_score = cls_score.sigmoid()
            num_sample = cls_score.shape[0]
            pos_loss, bag_acc, num_pos = self.loss_mil2(
                cls_score,
                ins_score,
                labels,
                label_valid,
                label_weights.unsqueeze(-1), )
            if isinstance(pos_loss, dict):
                losses.update(pos_loss)
            else:
                losses['loss_instance_mil'] = pos_loss
            losses['bag_acc'] = bag_acc

        if neg_cls_score is not None:
            num_neg, num_class = neg_cls_score.shape
            neg_cls_score = neg_cls_score.sigmoid()
            neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
            loss_weights = 0.75
            neg_valid = neg_weights.reshape(num_neg, -1)
            assert num_sample != 0
            neg_loss = self.loss_mil2.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
            neg_loss = loss_weights * label_weights.float().mean() * weight_reduce_loss(neg_loss, None,
                                                                                        avg_factor=num_sample)
            losses.update({"neg_loss": neg_loss})
        
        return losses
    
    def forward_train(self,
                      stage,
                      x,
                      img_metas,
                      proposal_list_base,
                      proposals_list,
                      proposals_valid_list,
                      neg_proposal_list,
                      neg_weight_list,
                      gt_points,
                      gt_labels,
                      dynamic_weight,
                      teacher=True,
                      gt_points_ignore=None,
                      gt_masks=None,
                      ):

        losses = dict()
        if teacher:
            bbox_results = self._bbox_forward_train_teacher(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                            neg_proposal_list,
                                                            neg_weight_list,
                                                            gt_points, gt_labels, dynamic_weight,
                                                            img_metas, stage)
            return losses, bbox_results['pseudo_boxes'], bbox_results['dynamic_weight']
        else:
            bbox_results = self._bbox_forward_train_student(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                            neg_proposal_list,
                                                            neg_weight_list,
                                                            gt_points, gt_labels, dynamic_weight,
                                                            img_metas, stage)
            losses.update(bbox_results['loss_instance_mil'])   
            return losses
    
    def _bbox_forward_train_teacher(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                                    neg_weight_list, gt_points,
                                    gt_labels,
                                    cascade_weight,
                                    img_metas, stage):
        rois = rbbox2roi(proposals_list)
        bbox_results = self._bbox_forward(x, rois, gt_points, stage)
        del rois; torch.cuda.empty_cache()
        
        gt_labels = torch.cat(gt_labels)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)
        
        pseudo_boxes, _, _, dynamic_weight = self.merge_box(bbox_results,
                                                            proposals_list,
                                                            proposals_valid_list,
                                                            gt_labels,
                                                            gt_points,
                                                            img_metas, stage)
        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=dynamic_weight.sum(dim=-1))

        return bbox_results
    
    def _bbox_forward_train_student(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                                    neg_weight_list, gt_points,
                                    gt_labels,
                                    cascade_weight,
                                    img_metas, stage):
        rois = rbbox2roi(proposals_list)
        bbox_results = self._bbox_forward(x, rois, gt_points, stage)
        del rois; torch.cuda.empty_cache()
        
        gt_labels = torch.cat(gt_labels)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)

        if neg_proposal_list is not None:
            neg_rois = rbbox2roi(neg_proposal_list)
            neg_bbox_results = self._bbox_forward(x, neg_rois, None, stage)  ######stage
            del neg_rois; torch.cuda.empty_cache()
            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
        else:
            neg_cls_scores = None
            neg_weights = None

        loss_instance_mil = self.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
                                         proposals_valid_list,
                                         neg_cls_scores, neg_weights,
                                         None, gt_labels,
                                         torch.cat(proposal_list_base), label_weights=cascade_weight,
                                         retrain_weights=None)

        bbox_results.update(loss_instance_mil=loss_instance_mil)
        return bbox_results

    def _bbox_forward(self, x, rois, gt_points, stage):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        cls_score, ins_score, reg_box = self.forward_mil_head(bbox_feats, stage)
        
        del bbox_feats; torch.cuda.empty_cache()

        # positive sample
        if gt_points is not None:
            num_gt = torch.cat(gt_points).shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_points}'
            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])

            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, num_instance=num_gt)
            return bbox_results
        # negative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, num_instance=None)
            return bbox_results

    def merge_box_single(self, cls_score, ins_score, dynamic_weight, gt_point, gt_label, proposals, img_metas, stage):
        if stage < self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'
        elif stage == self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'

        proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 5)
        h, w, c = img_metas['img_shape']
        num_gt, num_gen = proposals.shape[:2]
        # proposals = proposals.reshape(-1,4)
        if merge_mode == 'weighted_cls_topk':
            cls_score_, idx = cls_score.topk(k=self.topk2, dim=1)
            weight = cls_score_.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx] * weight).sum(dim=1)
            return boxes, None, None

        if merge_mode == 'weighted_clsins_topk':
            if stage == 0:
                k = self.topk1
            else:
                k = self.topk2
            dynamic_weight_, idx = dynamic_weight.topk(k=k, dim=1)
            weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 5])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            boxes = (filtered_boxes * weight).sum(dim=1)
            h, w, _ = img_metas['img_shape']
            
            boxes[:, :4] = bbox_cxcywh_to_xyxy(boxes[:, :4])
            boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, w)
            boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, h)
            boxes[:, :4] = bbox_xyxy_to_cxcywh(boxes[:, :4])
            
            # print(weight.sum(dim=1))
            # print(boxes)
            filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   dynamic_weight=dynamic_weight_)

            return boxes, filtered_boxes, filtered_scores


    def merge_box(self, bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, img_metas, stage):
        cls_scores = bbox_results['cls_score']
        ins_scores = bbox_results['ins_score']
        num_instances = bbox_results['num_instance']
        if stage < 1:
            cls_scores = cls_scores.softmax(dim=-1)
        else:
            cls_scores = cls_scores.sigmoid()
        ins_scores = ins_scores.softmax(dim=-2) * proposals_valid_list
        ins_scores = F.normalize(ins_scores, dim=1, p=1)
        cls_scores = cls_scores * proposals_valid_list
        dynamic_weight = (cls_scores * ins_scores)
        dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
        cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
        ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]
        # split batch
        batch_gt = [len(b) for b in gt_bboxes]
        cls_scores = torch.split(cls_scores, batch_gt)
        ins_scores = torch.split(ins_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
        if not isinstance(proposals_list, list):
            proposals_list = torch.split(proposals_list, batch_gt)
        stage_ = [stage for _ in range(len(cls_scores))]
        boxes, filtered_boxes, filtered_scores = multi_apply(self.merge_box_single, cls_scores, ins_scores,
                                                             dynamic_weight_list,
                                                             gt_bboxes,
                                                             gt_labels,
                                                             proposals_list,
                                                             img_metas, stage_)
        
        del cls_scores
        del ins_scores
        del proposals_list
        del proposals_valid_list
        torch.cuda.empty_cache()

        pseudo_boxes = torch.cat(boxes).detach()

        pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
        return list(pseudo_boxes), list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()


