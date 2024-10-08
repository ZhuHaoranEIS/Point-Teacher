import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, build_assigner, bbox2roi, build_bbox_coder
from ..builder import HEADS, build_loss, build_roi_extractor
from .anchor_free_head import AnchorFreeHead

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import math
import random

from ...core.bbox.iou_calculators import bbox_overlaps
from ...core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmcv.cnn import ConvModule

from mmdet.models.utils import build_linear_layer
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses.utils import weight_reduce_loss

from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels

from ..detectors.syn_images_generator_v2 import MIL_gen_proposals_from_cfg


INF = 1e8


@HEADS.register_module()
class TS_P2BFCOSHead(AnchorFreeHead):
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
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 mil_stack_conv=1,
                 beta=0.25,
                 top_k=3,
                 num_stages=2,
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
                 loss_bbox_burn1=dict(type='DIoULoss', loss_weight=1.0),
                 loss_bbox_burn2=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.1),
                 loss_bbox_denosing=dict(type='DN_DIoULoss', loss_weight=1.0, hyper=0.3),
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
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.mil_stack_conv = mil_stack_conv
        self.num_stages = num_stages
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox_burn1,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
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

        ### mil head
        self.beta = beta
        self.topk = top_k  # 3
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.loss_mil_iou = build_loss(dict(type='CrossEntropyLoss',
                                        use_sigmoid=True,
                                        loss_weight=0.25))
        self.loss_mil_bbox = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.25))
        self.mil_bbox_decoder = build_bbox_coder(dict(type='DeltaXYWHBBoxCoder',
                                                      target_means=[.0, .0, .0, .0],
                                                      target_stds=[1.0, 1.0, 1.0, 1.0]))
        self.loss_bbox_denosing = build_loss(loss_bbox_denosing)
        self.loss_bbox_burn2 = build_loss(loss_bbox_burn2)
        self.smoothl1 = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
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
            
         ### mil head
        self.num_shared_fcs = 2
        self.with_avg_pool = False
        self.fc_out_channels = 1024
        self.conv_out_channels = self.feat_channels
        self.roi_feat_area = 7 * 7

        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        _, self.shared_fcs_refine, last_layer_dim_refine = \
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
        self.fc_reg = nn.ModuleList()
        self.fc_iou = nn.ModuleList()

        self.shared_fcs_bag = nn.ModuleList()
        self.shared_fcs_reg = nn.ModuleList()

        num_cls = self.num_classes
        for i in range(self.num_stages):
            _, shared_fcs_bag, _ = \
                self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
            self.shared_fcs_bag.append(shared_fcs_bag)
            _, shared_fcs_reg, _ = \
                self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
            self.shared_fcs_reg.append(shared_fcs_reg)

            self.fc_cls.append(build_linear_layer(
                               dict(type='Linear'),
                               in_features=self.cls_last_dim,
                               out_features=num_cls))
            self.fc_ins.append(build_linear_layer(
                               dict(type='Linear'),
                               in_features=self.ins_last_dim,
                               out_features=num_cls))
            self.fc_reg.append(build_linear_layer(
                               dict(type='Linear'),
                               in_features=self.cls_last_dim,
                               out_features=4))
            self.fc_iou.append(build_linear_layer(
                               dict(type='Linear'),
                               in_features=self.cls_last_dim,
                               out_features=1))
    
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
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        all_level_points = self.get_points(featmap_sizes, feats[0].dtype,
                                           feats[0].device)
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
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        B, _, H, W = x.shape
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, points

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 
                          'all_level_points'))
    def get_pseudo_bbox(self,
             cls_scores,
             bbox_preds,
             centernesses,
             all_level_points,
             gt_points,
             gt_labels,
             gt_bboxes,
             filter_scores,
             img_metas,
             img_list,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores, flatten_bbox_preds, flatten_centernesses = self.concat_per_img(num_imgs, cls_scores, bbox_preds, centernesses)

        pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds = self.gnerate_pseudo(all_level_points, gt_points, gt_labels, gt_bboxes, filter_scores, img_metas, img_list, 
                                                                                            flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
        return pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                  'all_level_points'))
    def loss_pseudo(self,
            cls_scores,
            bbox_preds,
            centernesses,
            all_level_points,
            gt_points,
            gt_labels,
            pseudo_points, 
            pseudo_labels,
            pseudo_bboxes,
            gt_augument_ignore,
            img_metas,
            img_list,
            burn_in_step1,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores_img, flatten_bbox_preds_img, flatten_centernesses_img = self.concat_per_img(num_imgs, cls_scores, bbox_preds, centernesses)

        labels_reg, bbox_targets, labels, weights = self.get_target_pseudo(all_level_points, gt_points, gt_labels, pseudo_bboxes, img_metas, gt_augument_ignore,
                                                                                 pseudo_points, pseudo_labels, img_list, 
                                                                                 flatten_cls_scores_img, flatten_bbox_preds_img, flatten_centernesses_img, burn_in_step1)
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_weights = torch.cat(weights)
        flatten_labels_reg = torch.cat(labels_reg)
        flatten_bbox_targets = torch.cat(bbox_targets)
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
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox_burn2(
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
        apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                  'all_level_points'))
    def loss(self,
            cls_scores,
            bbox_preds,
            centernesses,
            all_level_points,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores_img, flatten_bbox_preds_img, flatten_centernesses_img = self.concat_per_img(num_imgs, cls_scores, bbox_preds, centernesses)

        labels, bbox_targets = self.get_targets(
            all_level_points, gt_bboxes, flatten_cls_scores_img, flatten_bbox_preds_img, flatten_centernesses_img)
        
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
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
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
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
    
    def get_targets(self, points, gt_bboxes_list, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses):
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        concat_cls_scores, concat_bbox_preds, concat_centernesses = self.concat_prediction(num_levels, flatten_cls_scores, flatten_bbox_preds, flatten_centernesses)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
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

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets)
    
    def _get_target_single(self, gt_bboxes, cls_scores, bbox_preds, centerness, points,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_labels = torch.zeros(num_gts).to(gt_bboxes.device).long()
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
        
        gt_bboxes_assigner = bbox_xyxy_to_cxcywh(gt_bboxes)
        assign_result = self.syn_assigner.assign(points, cls_scores, 
                                            gt_bboxes_assigner, gt_labels, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        labels = self.num_classes * torch.ones(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]
        
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        
        inds = inds * 0 
        inds[pos_inds] = assign_result.gt_inds[pos_inds] - 1
        bbox_targets = bbox_targets[range(num_points), inds]

        return labels, bbox_targets
    
    def get_target_pseudo(self, points, gt_points_list, gt_labels_list, pseudo_bboxes, img_metas, gt_augument_ignore, 
                          pseudo_points, pseudo_labels, img_list, 
                          flatten_cls_scores, flatten_bbox_preds, flatten_centernesses, burn_in_step1):
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
                img_list,
                gt_augument_ignore,
                points=concat_points,
                num_points_per_lvl=num_points,
                burn_in_step1=burn_in_step1)
        
        labels_reg_list, bbox_targets_list, labels_list, weights_list = results

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        weights_list = [weights.split(num_points, 0) for weights in weights_list]
        labels_reg_list = [labels.split(num_points, 0) for labels in labels_reg_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_weights = []
        concat_lvl_labels_reg = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_weights.append(
                torch.cat([weights[i] for weights in weights_list]))
            concat_lvl_labels_reg.append(
                torch.cat([labels[i] for labels in labels_reg_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
        return (concat_lvl_labels_reg, concat_lvl_bbox_targets, concat_lvl_labels, concat_lvl_weights)
    
    def _get_target_pseudo_single(self, gt_points, gt_labels, 
                                  pseudo_points, pseudo_labels, pseudo_bboxes, 
                                  cls_scores, bbox_preds, centernesses, 
                                  img_metas, img_list, gt_augument_ignore, points, num_points_per_lvl, burn_in_step1):
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
        num_gts = len(pseudo_bboxes)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   pseudo_bboxes.new_zeros((num_points, 4)), \
                   labels, \
                   weights
        
        pseudo_bboxes_assigner = bbox_xyxy_to_cxcywh(pseudo_bboxes)
        assign_result = self.pseudo_assigner.assign(points, cls_scores, 
                                                    pseudo_bboxes_assigner, pseudo_labels, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)

        labels_reg = self.num_classes * torch.ones(num_points, dtype=torch.long, device=pseudo_bboxes.device)
        labels_reg[pos_inds] = assign_result.labels[pos_inds]

        pseudo_bboxes = pseudo_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        
        left = xs - pseudo_bboxes[..., 0]
        right = pseudo_bboxes[..., 2] - xs
        top = ys - pseudo_bboxes[..., 1]
        bottom = pseudo_bboxes[..., 3] - ys
        
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        
        inds = inds * 0 
        inds[pos_inds] = assign_result.gt_inds[pos_inds] - 1
        bbox_targets = bbox_targets[range(num_points), inds]

        return labels_reg, bbox_targets, labels, weights
    
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
            return torch.empty((0, 4), device=gt_labels.device), \
                   torch.empty((0, 2), device=gt_labels.device), \
                   torch.empty((0, 1), device=gt_labels.device), \
                   0.0, \
                   None
        
        cls_scores_activate = cls_scores.clone().detach().sigmoid()
        bbox_pred_used = bbox_preds.clone().detach()
        bbox_preds_decode = distance2bbox(points, bbox_pred_used)
        bbox_preds_cxcywh = bbox_xyxy_to_cxcywh(bbox_preds_decode)
        assign_result = self.fuse_assigner.assign(bbox_preds_cxcywh, points, cls_scores, centernesses,
                                                    gt_points, gt_labels, gt_bboxes_ignore=None)
        
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)

        _, sorted_indices = torch.sort(inds[pos_inds]-1)
        pos_inds = pos_inds[sorted_indices]
        
        labels = torch.zeros(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]
            
        cls_scores_activate = cls_scores_activate[range(num_points), labels].reshape(-1,1)

        A = bbox_preds_decode[pos_inds,:]
        B = assign_result.gt_inds[pos_inds] - 1
        C = cls_scores_activate[pos_inds,:].reshape(-1)
        
        assign_nums = torch.bincount(B, minlength=num_gts)
        pseudo_bboxes = 8 * torch.ones_like(gt_bboxes)
        pseudo_scores = torch.zeros(num_gts).to(cls_scores_activate.device).to(cls_scores_activate.dtype)
        pseudo_points = gt_points.clone()
        pseudo_bboxes[:,:2] = pseudo_points
        pseudo_bboxes = bbox_cxcywh_to_xyxy(pseudo_bboxes)
        
        B_one_hot = torch.nn.functional.one_hot(B, num_classes=num_gts) # M * num_gts
        
        pseudo_bboxes_sum = torch.matmul(B_one_hot.T.float(), A * C.unsqueeze(1))  # num_gts * 5
        pseudo_scores_sum = torch.matmul(B_one_hot.T.float(), C)   # num_gts * 1
        
        zero_inds = (assign_nums != 0).nonzero().reshape(-1)
        
        pseudo_bboxes[zero_inds,:] = pseudo_bboxes_sum[zero_inds,:] / pseudo_scores_sum[zero_inds].unsqueeze(1)
        pseudo_scores[zero_inds] = pseudo_scores_sum[zero_inds] / assign_nums[zero_inds]
        pseudo_points[zero_inds,:] = bbox_xyxy_to_cxcywh(pseudo_bboxes[zero_inds,:])[:,:2]
                
        mean_ious_pred = bbox_overlaps(pseudo_bboxes[zero_inds,:], gt_bboxes[zero_inds,:], mode='iou', is_aligned=True).mean()

        filter_inds = (pseudo_scores >= filter_scores).nonzero().reshape(-1)

        valid_inds = torch.Tensor(list(set(zero_inds.tolist()) & set(filter_inds.tolist()))).long().to(zero_inds.device)
        pseudo_labels = gt_labels
        
        return pseudo_bboxes, pseudo_points, pseudo_labels, mean_ious_pred, valid_inds

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'all_level_points'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   all_level_points,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

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
    

    def concat_per_img(self, num_imgs, cls_scores, bbox_preds, centernesses):
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centernesses = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for centerness in centernesses
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
    
    def gfocal_loss(self, p, q, w=1.0, eps=1e-6):
        l1 = (p - q) ** 2
        # l1 = 1.0
        l2 = q * (p + eps).log() + (1 - q) * (1 - p + eps).log()
        return -(l1 * l2 * w).sum(dim=-1)

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
    
    def mil_bag_selection_single(self, cls_score, ins_score, proposals, img_metas, pseudo_bboxes):
        proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 4)
        h, w, c = img_metas['img_shape']
        selected_scores = cls_score * ins_score
        
        scores, idx = selected_scores.topk(k=self.topk, dim=1)
        weight = scores.unsqueeze(2).repeat([1, 1, 4])
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        # print(weight)
        filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
        boxes = (filtered_boxes * weight).sum(dim=1)
        h, w, _ = img_metas['img_shape']
        
        boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
        boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
        boxes = boxes.detach()

        boxes = (1-self.beta) * boxes + self.beta * pseudo_bboxes
        return boxes, scores

    def mil_bag_selection(self, bbox_results, img_metas, pseudo_bboxes, pseudo_labels):
        ### cls score: (num_gt, bbox_results['base_shaking_num'], -1, cls_score.shape[-1])
        pseudo_labels = torch.cat(pseudo_labels)
        cls_score = bbox_results['cls_score'].clone().detach()
        ins_score = bbox_results['ins_score'].clone().detach()

        num_gt, U1, U2, num_classes = cls_score.shape
        extensive_proposal_bag_list = bbox_results['extensive_bags']
        extensive_proposal_bag_valid_list = bbox_results['extensive_bags_valid']
        extensive_proposal_bag_valid_list = torch.cat(extensive_proposal_bag_valid_list, dim=0).reshape(
                    num_gt, U1*U2, 1)
        extensive_proposal_bag_list = torch.cat(extensive_proposal_bag_list, dim=0).reshape(
                    num_gt, U1*U2, 4)
        
        cls_score = cls_score.reshape(num_gt, U1*U2, num_classes).sigmoid()
        # cls_score = cls_score.reshape(num_gt, U1*U2, num_classes).softmax(dim=-1)
        ins_score = ins_score.softmax(dim=2)
        ins_score = ins_score * (extensive_proposal_bag_valid_list.reshape(num_gt, U1, U2, 1))
        ins_score = F.normalize(ins_score, dim=2, p=1)
        ins_score = ins_score.reshape(num_gt, U1*U2, num_classes)

        cls_score = cls_score[torch.arange(num_gt), :, pseudo_labels].reshape(num_gt, U1*U2)
        ins_score = ins_score[torch.arange(num_gt), :, pseudo_labels].reshape(num_gt, U1*U2)
        extensive_proposal_bag_list = extensive_proposal_bag_list[torch.arange(num_gt), :, :].reshape(num_gt, U1*U2, 4)

        batch_gt = [len(b) for b in pseudo_bboxes]
        cls_score = torch.split(cls_score, batch_gt)
        ins_score = torch.split(ins_score, batch_gt)
        extensive_proposal_bag_list = torch.split(extensive_proposal_bag_list, batch_gt)

        ### merge proposals
        merged_proposals, _ = multi_apply(self.mil_bag_selection_single, cls_score, ins_score, 
                                          extensive_proposal_bag_list, img_metas, pseudo_bboxes)
        return merged_proposals
    
    def mil_bag_training(self, bbox_results, gt_labels, neg_weight_list):
        cls_score = bbox_results['cls_score']; ins_score = bbox_results['ins_score']; iou_target = bbox_results['iou_target']
        proposals_valid_list = bbox_results['extensive_bags_valid']

        num_gt, U1, U2, num_classes = cls_score.shape
        labels = torch.cat(gt_labels).unsqueeze(dim=1).repeat(1, U1).reshape(-1) # num_gt * U1
        proposals_valid_list = torch.cat(proposals_valid_list, dim=0).reshape(
            num_gt, U1, U2, 1)
        
        cls_score = cls_score.sigmoid()
        # cls_score = cls_score.softmax(dim=-1)
        ins_score = ins_score.softmax(dim=2)
        ins_score = ins_score * proposals_valid_list
        ins_score = F.normalize(ins_score, dim=2, p=1)
        pos_bag_score = (cls_score * ins_score).sum(dim=2).reshape(-1, num_classes)
        
        ### bag loss
        label_weights = (proposals_valid_list.reshape(num_gt*U1, U2, 1).sum(dim=1) > 0).float()
        num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)
        labels = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
        loss_pos_bag = self.gfocal_loss(pos_bag_score, labels, label_weights)
        loss_mil_bags = weight_reduce_loss(loss_pos_bag, None, avg_factor=num_sample)
        if neg_weight_list != None:
            neg_cls_score = bbox_results['neg_cls_score']
            neg_cls_score = neg_cls_score.sigmoid()
            # neg_cls_score = neg_cls_score.softmax(dim=-1)
            neg_bag_score = neg_cls_score
            num_neg, num_class = neg_bag_score.shape
            neg_labels = torch.full((num_neg, num_class), 0, dtype=neg_bag_score.dtype).to(neg_bag_score.device)
            neg_weights = torch.cat(neg_weight_list)
            neg_valid = neg_weights.reshape(num_neg, -1)
            loss_neg_bag = self.gfocal_loss(neg_bag_score, neg_labels, neg_valid.float())
            loss_mil_bags += weight_reduce_loss(loss_neg_bag, None, avg_factor=num_sample)
        return loss_mil_bags
    
    def mil_bag_extensive(self, num_gt, num_gt_pre_image, x, img_metas, proposals_list, proposals_valid_list, 
                          proposals_reference_list, proposals_real_list, bbox_results, fine_proposal_cfg, stage):
        ### bags extensive
        U1 = int(proposals_list[0].shape[0] / num_gt_pre_image[0])
        bbox_results['base_shaking_num'] = U1

        points_list = [bbox_xyxy_to_cxcywh(proposals_list[i])[:,:2] for i in range(len(proposals_list))]
        extensive_proposal_bags_list, extensive_proposal_bags_valid_list, _, extensive_proposal_bags_reference_list = \
            MIL_gen_proposals_from_cfg(points_list, proposals_list, fine_proposal_cfg, proposals_reference_list, img_metas)
        _, _, _, extensive_proposal_bags_real_list = \
            MIL_gen_proposals_from_cfg(points_list, proposals_list, fine_proposal_cfg, proposals_real_list, img_metas)
        
        bbox_results['base_bags'] = proposals_list # len = bs: [(num_gt_i*U1, 4), ...]
        bbox_results['base_bags_valid'] = proposals_valid_list
        bbox_results['coarse_bags_iou'] = bbox_overlaps(torch.cat(extensive_proposal_bags_list), torch.cat(extensive_proposal_bags_real_list), 
                                                        mode='iou', is_aligned=True).mean()
        U2 = int(extensive_proposal_bags_list[0].shape[0] / (num_gt_pre_image[0]*U1))
        bbox_results['extensive_shaking_num'] = U2
        ### bags refinement
        rois = bbox2roi(extensive_proposal_bags_list)
        proposal_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        proposal_feats = proposal_feats.flatten(1)
        for fc in self.shared_fcs_reg[stage]:
            proposal_feats = self.relu(fc(proposal_feats))
        bbox_pred = self.fc_reg[stage](proposal_feats) # L * 4
        del proposal_feats; torch.cuda.empty_cache()

        bbox_pred = self.mil_bbox_decoder.decode(
            torch.cat(extensive_proposal_bags_list), bbox_pred, max_shape=img_metas[0]['img_shape'])
        bbox_pred_clone = bbox_pred.clone().detach()
        bbox_target = torch.cat(extensive_proposal_bags_reference_list)

        bbox_results['loss_mil_bbox'] = self.loss_bbox_denosing(bbox_pred, 
                                                                bbox_target, 
                                                                weight=torch.cat(extensive_proposal_bags_valid_list).reshape(-1).float(), 
                                                                avg_factor=bbox_pred.shape[0])
        bbox_results['refine_bags_iou'] = bbox_overlaps(bbox_pred_clone, torch.cat(extensive_proposal_bags_real_list), 
                                                        mode='iou', is_aligned=True).mean()
        
        bbox_results['iou_target'] = bbox_overlaps(bbox_pred_clone, torch.cat(extensive_proposal_bags_reference_list), 
                                                   mode='iou', is_aligned=True).reshape(-1)

        refined_extensive_proposal_bags_list = []
        index = 0
        for i in range(len(extensive_proposal_bags_list)):
            num = extensive_proposal_bags_list[i].shape[0]
            extensive_proposal_bags_gti = bbox_pred_clone[index:(index+num),:]
            refined_extensive_proposal_bags_list.append(extensive_proposal_bags_gti)
            index += num

        bbox_results['extensive_bags'] = refined_extensive_proposal_bags_list # len = bs: [(num_gt_i*U1*U2, 4), ...]
        bbox_results['extensive_bags_valid'] = extensive_proposal_bags_valid_list
        bbox_results['extensive_bags_reference'] = extensive_proposal_bags_reference_list
        bbox_results['extensive_bags_real'] = extensive_proposal_bags_real_list



    def mil_bag_classifier(self, num_gt, x, bbox_results, stage):
        extensive_bag_proposal_list = bbox_results['extensive_bags']
        extensive_rois = bbox2roi(extensive_bag_proposal_list)
        extensive_proposal_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], extensive_rois)
        extensive_proposal_feats = extensive_proposal_feats.flatten(1)
        for fc in self.shared_fcs_bag[stage]:
            extensive_proposal_feats = self.relu(fc(extensive_proposal_feats))
        cls_feat = extensive_proposal_feats; ins_feat = extensive_proposal_feats
        cls_score = self.fc_cls[stage](cls_feat); ins_score = self.fc_ins[stage](ins_feat)
        del extensive_proposal_feats; del cls_feat; del ins_feat; torch.cuda.empty_cache()
        
        cls_score = cls_score.view(num_gt, bbox_results['base_shaking_num'], bbox_results['extensive_shaking_num'], cls_score.shape[-1])
        ins_score = ins_score.view(num_gt, bbox_results['base_shaking_num'], bbox_results['extensive_shaking_num'], ins_score.shape[-1])

        bbox_results['cls_score'] = cls_score
        bbox_results['ins_score'] = ins_score


    def forward_mil_head(self, num_gt, num_gt_pre_image, x, proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list, 
                         img_metas, fine_proposal_cfg, stage, neg_proposal_list=None, neg_weight_list=None):
        bbox_results = {}
        self.mil_bag_extensive(num_gt, num_gt_pre_image, x, img_metas, proposals_list, proposals_valid_list,
                               proposals_reference_list, proposals_real_list, bbox_results, fine_proposal_cfg, stage)
        self.mil_bag_classifier(num_gt, x, bbox_results, stage)

        if neg_proposal_list != None:
            neg_rois = bbox2roi(neg_proposal_list)
            neg_proposal_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], neg_rois)
            neg_proposal_feats = neg_proposal_feats.flatten(1)
            for fc in self.shared_fcs_bag[stage]:
                neg_proposal_feats = self.relu(fc(neg_proposal_feats))
            neg_cls_score = self.fc_cls[stage](neg_proposal_feats)
            del neg_proposal_feats; torch.cuda.empty_cache()
            bbox_results['neg_cls_score'] = neg_cls_score

        return bbox_results
        
    def MIL_head_burn_in_step1(self,
                               x_ori,
                               x_synethic,
                               img_metas,
                               proposals_list,
                               proposals_valid_list,
                               proposals_reference_list,
                               proposals_real_list,
                               syn_proposals_list,
                               syn_proposals_valid_list,
                               syn_proposals_reference_list,
                               syn_proposals_real_list,
                               neg_proposal_list,
                               neg_weight_list,
                               synthetic_bboxes,
                               pseudo_bboxes,
                               pseudo_labels,
                               fine_proposal_cfg,
                               stage):
        num_synethic = torch.cat(synthetic_bboxes).shape[0]; num_synthetic_pre_image = [synthetic_bboxes[i].shape[0] for i in range(len(synthetic_bboxes))]
        num_gt = torch.cat(pseudo_bboxes).shape[0]; num_gt_pre_image = [pseudo_bboxes[i].shape[0] for i in range(len(pseudo_bboxes))]
        losses = {}
        syn_bbox_results = self.forward_mil_head(num_synethic, num_synthetic_pre_image, x_synethic, syn_proposals_list, syn_proposals_valid_list, syn_proposals_reference_list, 
                                                 syn_proposals_real_list, img_metas, fine_proposal_cfg, stage=stage, neg_proposal_list=None, neg_weight_list=None)
        del x_synethic; torch.cuda.empty_cache()
        losses[f'stage{stage}_loss_mil_bbox'] = syn_bbox_results['loss_mil_bbox']
        del syn_bbox_results; torch.cuda.empty_cache()

        bbox_results = self.forward_mil_head(num_gt, num_gt_pre_image, x_ori, proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list,
                                             img_metas, fine_proposal_cfg, stage=stage, neg_proposal_list=neg_proposal_list, neg_weight_list=neg_weight_list)
        loss_mil_bags = self.mil_bag_training(bbox_results, pseudo_labels, neg_weight_list)
        losses[f'stage{stage}_loss_mil_bags'] = loss_mil_bags
        losses[f'stage{stage}_coarse_bags_iou'] = bbox_results['coarse_bags_iou']
        losses[f'stage{stage}_refine_bags_iou'] = bbox_results['refine_bags_iou']

        merged_proposals = self.mil_bag_selection(bbox_results, img_metas, pseudo_bboxes, pseudo_labels)

        return losses, merged_proposals

    def MIL_head_burn_in_step2(self,
                               x,
                               img_metas,
                               proposals_list,
                               proposals_valid_list,
                               proposals_reference_list,
                               proposals_real_list,
                               neg_proposal_list,
                               neg_weight_list,
                               pseudo_bboxes,
                               pseudo_labels,
                               fine_proposal_cfg,
                               stage):
        num_gt = torch.cat(pseudo_bboxes).shape[0]
        num_gt_pre_image = [pseudo_bboxes[i].shape[0] for i in range(len(pseudo_bboxes))]
        losses = {}
        bbox_results = self.forward_mil_head(num_gt, num_gt_pre_image, x, proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list,
                                             img_metas, fine_proposal_cfg, stage=stage, neg_proposal_list=neg_proposal_list, neg_weight_list=neg_weight_list)
        losses[f'stage{stage}_loss_mil_bbox'] = bbox_results['loss_mil_bbox']
        
        loss_mil_bags = self.mil_bag_training(bbox_results, pseudo_labels, neg_weight_list)
        losses[f'stage{stage}_loss_mil_bags'] = loss_mil_bags
        losses[f'stage{stage}_coarse_bags_iou'] = bbox_results['coarse_bags_iou']
        losses[f'stage{stage}_refine_bags_iou'] = bbox_results['refine_bags_iou']

        merged_proposals = self.mil_bag_selection(bbox_results, img_metas, pseudo_bboxes, pseudo_labels)
        return losses, merged_proposals
        
    def inference_mil_head(self,
                           x,
                           img_metas,
                           proposals_list,
                           proposals_valid_list,
                           proposals_reference_list,
                           proposals_real_list,
                           pseudo_bboxes,
                           pseudo_labels,
                           fine_proposal_cfg,
                           stage):
        losses = {}
        num_gt = torch.cat(pseudo_bboxes).shape[0]
        num_gt_pre_image = [pseudo_bboxes[i].shape[0] for i in range(len(pseudo_bboxes))]
        if fine_proposal_cfg != None:
            bbox_results = self.forward_mil_head(num_gt, num_gt_pre_image, x, proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list,
                                                img_metas, fine_proposal_cfg, stage=stage, neg_proposal_list=None, neg_weight_list=None)
            losses[f'stage{stage}_coarse_bags_iou'] = bbox_results['coarse_bags_iou']
            losses[f'stage{stage}_refine_bags_iou'] = bbox_results['refine_bags_iou']
        else:
            bbox_results = {}
            rois = bbox2roi(proposals_list)
            proposal_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            proposal_feats = proposal_feats.flatten(1)
            for fc in self.shared_fcs:
                proposal_feats = self.relu(fc(proposal_feats))
            cls_feat = proposal_feats; ins_feat = proposal_feats
            cls_score = self.fc_cls(cls_feat); ins_score = self.fc_ins(ins_feat)
            del proposal_feats; del cls_feat; del ins_feat; torch.cuda.empty_cache()

            cls_score = cls_score.view(num_gt, 1, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, 1, -1, ins_score.shape[-1])
            
            bbox_results['cls_score'] = cls_score
            bbox_results['ins_score'] = ins_score
            bbox_results['extensive_bags'] = proposals_list
            bbox_results['extensive_bags_valid'] = proposals_valid_list

            coarse_bags_iou = bbox_overlaps(torch.cat(proposals_list), torch.cat(proposals_real_list), mode='iou', is_aligned=True).mean()
            losses[f'stage{stage}_coarse_bags_iou'] = coarse_bags_iou
            losses[f'stage{stage}_refine_bags_iou'] = coarse_bags_iou
        
        merged_proposals = self.mil_bag_selection(bbox_results, img_metas, pseudo_bboxes, pseudo_labels)
        return list(merged_proposals), losses
    