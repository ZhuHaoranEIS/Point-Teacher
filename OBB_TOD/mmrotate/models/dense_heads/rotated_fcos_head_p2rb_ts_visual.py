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
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_objectness = build_loss(dict(type='L1Loss', loss_weight=1.0))
        if self.separate_angle:
            self.loss_angle = build_loss(loss_angle)
            self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
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
            
            
        ### dict_cls_scores
        self.dict_image_flag = {}
        self.dict_last_cls_scores = {}

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_mil = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)


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
            bbox_pred = bbox_pred.clamp(min=0)
            bbox_pred *= stride
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

        PALETTENOISE = [(0, 255, 0), (0,  0, 255), (255, 144, 30), (0, 173, 205), (128, 128, 0)]
        image_path = img_metas['ori_filename']
        image_store_path = "/home/zhr/SODA/mmrotate/models/detectors/syn"
        image_store_path = os.path.join(image_store_path, image_path)
        img_plot = img_syn.permute(1, 2, 0).contiguous().cpu().numpy()
        cv2.imwrite(image_store_path, img_plot)
        image = cv2.imread(image_store_path)
        for j in range(len(pos_inds)):
            center_x, center_y = points[pos_inds[j],:]
            color = PALETTENOISE[0]
            radius = 2
            cv2.circle(image, (int(center_x), int(center_y)), radius, color, 2)
        cv2.imwrite(image_store_path, image)
        imshow_det_rbboxes(image_store_path,
                bboxes=gt_bboxes.cpu().numpy(),
                labels=np.ones(gt_bboxes.shape[0]).astype(int),
                bbox_color='green',
                text_color='green',
                out_file=image_store_path)

        imshow_det_rbboxes(image_store_path,
                bboxes=self.bbox_coder.decode(points, bbox_preds)[pos_inds,:].detach().cpu().numpy(),
                labels=np.ones(len(pos_inds)).astype(int),
                bbox_color='red',
                text_color='red',
                out_file=image_store_path)

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
        cls_scores_activate = cls_scores.clone().sigmoid().detach()
        weights = torch.ones_like(labels).float()
        if imagefile in self.dict_image_flag.keys():
            neg_inds = (inds == 0).nonzero(as_tuple=False).reshape(-1)
            neg_cls_pred = (cls_scores_activate[neg_inds,:]).max(1)[0]
            neg_cls_pred_last = self.dict_last_cls_scores[imagefile][neg_inds]
            weight = neg_cls_pred_last / neg_cls_pred

            inds_one = (weight >= 1).nonzero().reshape(-1)
            weight[inds_one] = 1.0
            weight = torch.where(torch.isnan(weight), torch.full_like(weight, 1.0), weight)
            weights[neg_inds] = weight
        ### store information
        self.dict_image_flag[imagefile] = 1
        self.dict_last_cls_scores[imagefile] = cls_scores_activate.max(1)[0]

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
        mil_feat = self.conv_mil(x)
        return mil_feat, x