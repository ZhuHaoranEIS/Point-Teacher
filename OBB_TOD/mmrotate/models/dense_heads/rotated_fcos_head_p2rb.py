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

import time
import cv2
import os
import numpy as np
import itertools
from mmcv.ops import DeformConv2d
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ..detectors.data_augument_bank import imshow_det_rbboxes

from torchvision import transforms
from ..detectors.rotated_fcos_p2rb import rbbox_augument

from PIL import Image
from torchvision.transforms import functional as F_resize

INF = 1e8


@ROTATED_HEADS.register_module()
class P2RBRotatedFCOSHead(RotatedAnchorFreeHead):
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
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 separate_angle=False,
                 scale_angle=True,
                 beta = 100,
                 frozen_iters = 1000,
                 temporal_threshold = 0.1,
                 filter_size=40,
                 w1=1.0,
                 h_bbox_coder=dict(type='DistancePointBBoxCoder'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_angle=dict(type='L1Loss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=dict(
                     assigner=dict(
                         type='TopkAssigner',
                         topk=3,
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='PointCost', mode='L1', weight=5.0)),
                     aug_assigner=dict(
                         type='BboxTopkAssigner',
                         topk=3,
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='PointCost', mode='L1', weight=1.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
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
        if self.separate_angle:
            self.loss_angle = build_loss(loss_angle)
            self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
        # Angle predict length
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)   
            aug_assigner = train_cfg['aug_assigner']
            self.aug_assigner = build_assigner(aug_assigner) 
            self.preds_decoder = build_bbox_coder(dict(type='DistanceAnglePointCoder'))

        ### DCM
        self.dcm = torch.zeros(1)
        self.beta = 1 - 1 / beta
        self.iters = 0
        self.frozen_iters = frozen_iters
        self.temporal_threshold = temporal_threshold
        self.filter_size = filter_size
        ### sl loss
        self.loss_dxdydwdhda = build_loss(dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0))
        ### dynamic gt bbox:
        self.gt_pred_bboxes_bank = {}
        self.gt_pred_scores_bank = {}
        self.w1 = w1

        self.transform = transforms.Compose([
            transforms.ToTensor()  # 转换为 PyTorch 张量，并将像素值缩放到 [0, 1] 范围
        ])



    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.relu = nn.ReLU()
        self.conv_xy = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.conv_wh = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_objectness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)
        
        self.conv_reg_wh_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)
        self.conv_reg_theta_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)



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
        points_ori = points.clone()
        points = points.unsqueeze(0).repeat(B,1,1).reshape(-1,2)
        cls_feat = x
        reg_feat = x

        cls_feat_bank = [cls_feat, ]
        reg_feat_bank = [reg_feat, ]
        for i in range(len(self.cls_convs)):
            cls_feat = self.cls_convs[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_feat)
            cls_feat_bank.append(cls_feat)
            reg_feat_bank.append(reg_feat)
        
        cls_score = self.conv_cls(cls_feat)
        objectness = self.conv_objectness(cls_feat)

        base_ltrb = stride
        base_size = stride * 2
        # cxcywh theta
        bbox_pred_decode = base_size * torch.ones(B,5,H,W).to(cls_score.dtype).to(cls_score.device)
        bbox_pred_decode[:,:2,:,:] = points.reshape(B,H,W,2).permute(0,3,1,2) # B 4 H W
        bbox_pred_decode[:,-1,:,:] = 0 * bbox_pred_decode[:,-1,:,:]
        ### first stage dxdy
        bbox_pred_dxdy = self.conv_xy(reg_feat).float().sigmoid() # B 2 H W
        bbox_pred_decode[:,:2,:,:] = torch.add(bbox_pred_dxdy * base_size, bbox_pred_decode[:,:2,:,:])
        reg_feat_theta_refine = self.conv_reg_theta_conv(reg_feat_bank[-2])
        ### second stage theta
        bbox_pred_theta = self.conv_theta(reg_feat_theta_refine) # B 1 H W
        if self.is_scale_angle:
            bbox_pred_theta = self.scale_angle(bbox_pred_theta).float()
        else:
            bbox_pred_theta = bbox_pred_theta.float()
        bbox_pred_decode[:,-1,:,:] = torch.add(bbox_pred_theta.squeeze(1), bbox_pred_decode[:,-1,:,:])
        # second stage dcn_offset get input ltrb
        reg_feat_wh_refine = self.conv_reg_wh_conv(reg_feat_bank[-2])
        # reg_feat_wh_refine = reg_feat_bank[-3]
        bbox_pred_dwdh = self.conv_wh(reg_feat_wh_refine) # B 2 H W
        bbox_pred_dwdh = bbox_pred_dwdh.float().exp()

        bbox_pred_decode[:,2:4,:,:] = base_size * torch.ones_like(bbox_pred_dwdh) * bbox_pred_dwdh

        if not self.training:
            return cls_score, bbox_pred_decode
        
        return cls_score, points_ori, objectness, bbox_pred_dxdy, bbox_pred_theta, bbox_pred_dwdh, bbox_pred_decode


    @force_fp32(
        apply_to=('cls_scores', 'all_level_points', 'objectness', 'bbox_preds_dxdy', 'bbox_preds_theta',
                  'bbox_preds_dwdh', 'bbox_preds_cxcywhtheta'))
    def loss_ori(self,
            cls_scores,
            all_level_points,
            objectness,
            bbox_preds_dxdy,
            bbox_preds_theta,
            bbox_preds_dwdh,
            bbox_preds_cxcywhtheta,
            gt_points,
            gt_labels,
            gt_bboxes_ref,
            img_list,
            img_metas,
            insider,
            gt_bboxes_ignore=None,
            augument_type=None, 
            augument_factor=None):
        assert len(cls_scores) == len(bbox_preds_cxcywhtheta) == len(objectness)
        if insider[0] != None:
            self.iters += 1
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores_img, flatten_bbox_preds_dxdy_img, flatten_bbox_preds_theta_img, flatten_bbox_preds_dwdh_img, \
            flatten_bbox_preds_cxcywhtheta_img, flatten_objectness_img = self.generate_per_img_all(num_imgs, cls_scores, bbox_preds_dxdy, 
                                                                                            bbox_preds_theta, bbox_preds_dwdh, bbox_preds_cxcywhtheta,
                                                                                            objectness)

        results = self.get_targets(all_level_points, gt_points, gt_labels, gt_bboxes_ref, insider, img_list, img_metas, 
                                                            flatten_cls_scores_img, flatten_bbox_preds_dxdy_img, flatten_bbox_preds_theta_img,
                                                            flatten_bbox_preds_dwdh_img, flatten_bbox_preds_cxcywhtheta_img, 
                                                            flatten_objectness_img, augument_type, augument_factor)
        
        labels, rbbox_preds_list, bbox_preds_dxdy_list, bbox_preds_theta_list, bbox_preds_dwdh_list, \
            gt_bboxes_insider_list, pos_gts, sam_overlap, labels_pseudo, bbox_targets, angle_targets = results
        
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_objectness = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, 1)
            for cls_score in objectness
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            for bbox_pred in bbox_preds_cxcywhtheta
        ]

        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels_pseudo = torch.cat(labels_pseudo)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_objectness = torch.cat(flatten_objectness)
        flatten_labels = torch.cat(labels)
        flatten_labels_objectness = torch.ones_like(flatten_labels)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        flatten_labels_objectness[pos_inds] = 0
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds_cxcywhtheta[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        loss_objectness = self.loss_cls(
            flatten_objectness, flatten_labels_objectness, avg_factor=num_pos)

        ### compute iou between prediction and GT, detach!!!
        _1st_preds = torch.cat(rbbox_preds_list).clone().detach()
        nan_gts = torch.cat(pos_gts)
        nonnan_inds = (nan_gts != 0).nonzero().reshape(-1)
        gt_bboxes_insider = torch.cat(gt_bboxes_insider_list)
        pred_rbbox_iou = rbbox_overlaps(_1st_preds[nonnan_inds], gt_bboxes_insider[nonnan_inds], mode='iou', is_aligned=True).nanmean()

        ### pseudo bbox regression
        pos_inds = ((flatten_labels_pseudo >= 0)
                    & (flatten_labels_pseudo < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds_cxcywhtheta[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.bbox_coder
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_target_decode = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            
            loss_bbox = self.loss_dxdydwdhda(pos_bbox_preds[:,:4],
                                             pos_decoded_target_decode[:,:4],
                                             avg_factor=num_pos) / 100
        else:
            loss_bbox = pos_bbox_preds.sum()
        
        return loss_cls, pred_rbbox_iou, rbbox_preds_list, bbox_preds_dxdy_list, bbox_preds_theta_list, bbox_preds_dwdh_list, \
               pos_gts, sam_overlap, loss_bbox, loss_objectness
        
    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_bboxes_ref_list, insider_list, img_list, img_metas, 
                    flatten_cls_scores, flatten_bbox_preds_dxdy, flatten_bbox_preds_theta, flatten_bbox_preds_dwdh, 
                    flatten_bbox_preds_cxcywhtheta, flatten_objectness, augument_type, augument_factor):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                    each level.
        """
        num_levels = len(points)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        concat_strides = []
        for i in range(num_levels):
            concat_strides.append(points[i].new_full((num_points[i],), self.strides[i]))
        concat_strides = torch.cat(concat_strides)
        concat_points = torch.cat(points, dim=0)
        concat_cls_scores, concat_bbox_preds_dxdy, concat_bbox_preds_theta, concat_bbox_preds_dwdh, concat_bbox_preds_cxcywhtheta, \
            concat_objectness = self.concat_per_img(num_levels, flatten_cls_scores, flatten_bbox_preds_dxdy, 
                                                    flatten_bbox_preds_theta, flatten_bbox_preds_dwdh,
                                                    flatten_bbox_preds_cxcywhtheta, flatten_objectness)
        
        # get labels and bbox_targets of each image
        results = multi_apply(
                self._get_targets_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ref_list,
                insider_list,
                concat_cls_scores,
                concat_bbox_preds_dxdy,
                concat_bbox_preds_theta,
                concat_bbox_preds_dwdh,
                concat_bbox_preds_cxcywhtheta,
                concat_objectness,
                img_list,
                img_metas,
                augument_type, 
                augument_factor,
                points=concat_points,
                strides=concat_strides,
                num_points_per_lvl=num_points)
        
        labels_list, rbbox_preds_list, bbox_preds_dxdy_list, bbox_preds_theta_list, bbox_preds_dwdh_list, \
        gt_bboxes_insider_list, pos_gts, sam_overlap, labels_pseudo_list, bbox_targets_list, angle_targets_list = results

        
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        labels_pseudo_list = [labels.split(num_points, 0) for labels in labels_pseudo_list]
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
        concat_lvl_labels_pseudo = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
    
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_labels_pseudo.append(
                torch.cat([labels[i] for labels in labels_pseudo_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        num_img = len(sam_overlap)
        return concat_lvl_labels, rbbox_preds_list, bbox_preds_dxdy_list, bbox_preds_theta_list, \
               bbox_preds_dwdh_list, gt_bboxes_insider_list, pos_gts, sum(sam_overlap)/num_img, \
               concat_lvl_labels_pseudo, concat_lvl_bbox_targets, concat_lvl_angle_targets
    
    def _get_targets_single(self, gt_bboxes, gt_labels, gt_bboxes_ref, insider_list, cls_scores, bbox_preds_dxdy, bbox_preds_theta,
                            bbox_preds_dwdh, bbox_preds_cxcywhtheta, objectness, 
                            img_list, img_metas, augument_type, augument_factor, 
                            points, strides, num_points_per_lvl):
        image_name = img_metas['ori_filename']
        num_points = points.size(0)
        if insider_list is not None:
            insider_inds = (insider_list == 0).nonzero(as_tuple=False).reshape(-1)
            gt_labels = gt_labels[insider_inds]
            gt_bboxes = gt_bboxes[insider_inds]
            gt_bboxes_ref = gt_bboxes_ref[insider_inds]
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes)
        gt_ori = gt_bboxes.clone()
        points_ori = points.clone()
        # label assignment
        if insider_list is None:
            assign_result = self.assigner.assign(points_ori, cls_scores.clone().detach(), 
                                                    gt_ori, gt_labels, gt_bboxes_ignore=None)
        else:
            gt_store_bboxes = self.gt_pred_bboxes_bank[image_name][insider_inds,:] # num_gts 5
            gt_store_bboxes = rbbox_augument(img_list, gt_store_bboxes, augument_factor, torch.zeros(num_gts).long().to(cls_scores.device), augument_type, 0)
            assign_result = self.aug_assigner.assign(bbox_preds_cxcywhtheta.clone().detach(), cls_scores.clone().detach(), 
                                                    gt_store_bboxes, gt_labels, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds
        pos_inds = (inds != 0).nonzero(as_tuple=False).reshape(-1)
        labels = self.num_classes * torch.ones(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels[pos_inds] = assign_result.labels[pos_inds]

        ### Compute average output for valid positions
        pos_inds_matrix = (inds.unsqueeze(0) == torch.arange(1, num_gts + 1, device=inds.device).unsqueeze(1))
        pos_valid_mask = pos_inds_matrix.any(dim=1)

        valid_gt_indices, valid_point_indices = pos_inds_matrix.nonzero(as_tuple=False).t()

        # Get pos_valid_inds and pos_gts
        pos_valid_inds = valid_point_indices
        pos_gts = pos_valid_mask.long()
        
        # Sum and average rbbox predictions
        rbbox_preds_sum = torch.zeros((num_gts, 5), device=gt_labels.device)
        counts = torch.zeros(num_gts, device=gt_labels.device)
        for i in range(bbox_preds_cxcywhtheta.shape[1]):
            rbbox_preds_sum[:, i].index_add_(0, valid_gt_indices, bbox_preds_cxcywhtheta[valid_point_indices][:, i])
        counts.index_add_(0, valid_gt_indices, torch.ones_like(valid_gt_indices, dtype=torch.float))
        rbbox_preds_average = rbbox_preds_sum / counts[:, None]

        # Sum and average bbox dxdy
        bbox_preds_dxdy_sum = torch.zeros((num_gts, bbox_preds_dxdy.shape[1]), device=gt_labels.device)
        for i in range(bbox_preds_dxdy.shape[1]):
            bbox_preds_dxdy_sum[:, i].index_add_(0, valid_gt_indices, bbox_preds_dxdy[valid_point_indices][:, i])
        bbox_preds_dxdy_average = bbox_preds_dxdy_sum / counts[:, None]

        # Sum and average bbox theta
        bbox_preds_theta_sum = torch.zeros((num_gts, bbox_preds_theta.shape[1]), device=gt_labels.device)
        for i in range(bbox_preds_theta.shape[1]):
            bbox_preds_theta_sum[:, i].index_add_(0, valid_gt_indices, bbox_preds_theta[valid_point_indices][:, i])
        bbox_preds_theta_average = bbox_preds_theta_sum / counts[:, None]

        # Sum and average bbox exp(dw, dh)
        bbox_preds_dwdh_sum = torch.zeros((num_gts, bbox_preds_dwdh.shape[1]), device=gt_labels.device)
        for i in range(bbox_preds_dwdh.shape[1]):
            bbox_preds_dwdh_sum[:, i].index_add_(0, valid_gt_indices, bbox_preds_dwdh[valid_point_indices][:, i])
        bbox_preds_dwdh_average = bbox_preds_dwdh_sum / counts[:, None]

        # Sum and average bbox pred scores
        cls_scores_labels = torch.zeros_like(labels)
        cls_scores_labels[pos_inds] = labels[pos_inds]
        cls_scores_refine = (cls_scores.clone().detach().sigmoid())[range(num_points),cls_scores_labels].reshape(-1,1)
        cls_scores_sum = torch.zeros((num_gts, 1), device=gt_labels.device)
        for i in range(cls_scores_sum.shape[1]):
            cls_scores_sum[:, i].index_add_(0, valid_gt_indices, cls_scores_refine[valid_point_indices][:, i])
        cls_scores_average = (cls_scores_sum / counts[:, None]).reshape(-1,1)
        
        ### update DCM
        self.dcm = self.dcm.to(cls_scores.device).to(cls_scores.dtype)
        pos_cls_scores = objectness[pos_inds].clone().detach().sigmoid().mean()
        self.dcm = self.beta * self.dcm + (1 - self.beta) * pos_cls_scores

        if insider_list is None:
            # ori view
            ### semantic response
            H, W = img_metas['pad_shape'][:2]
            semantic_response = torch.zeros(1, H, W).to(cls_scores.device).to(cls_scores.dtype) # 1 H W

            # get discriminative thresholds
            if self.iters >= self.frozen_iters:
                discriminative_thresholds = self.dcm
            else:
                discriminative_thresholds = self.temporal_threshold
            sam_pos_inds = (objectness.sigmoid() >= discriminative_thresholds).nonzero().reshape(-1)
            semantic_response[:,points[sam_pos_inds,:][:,0].long(),points[sam_pos_inds,:][:,1].long()] = 1
            semantic_response[:,points[pos_inds,:][:,0].long(),points[pos_inds,:][:,1].long()] = 1

            radius = int(self.strides[0]/4) ### important!!!!!
            kernel_size = int(2 * radius + 1)
            padding = radius
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=semantic_response.device)
            expanded_mask = F.conv2d(semantic_response.float().unsqueeze(0), kernel, padding=padding)
            semantic_response = (expanded_mask > 0).squeeze(0)
            semantic_response = semantic_response.float()

            ### cluster and find the rectangles
            mask_points = torch.nonzero(semantic_response.squeeze(0), as_tuple=False)
            distances = torch.cdist(mask_points.float(), gt_bboxes, p=2)
            min_indices = torch.argmin(distances, dim=1)

            # Filter points within filter_size/2 distance to their nearest bounding box
            filter_condition = distances[range(distances.shape[0]), min_indices] <= (self.filter_size / 1)
            # filter_condition = distances[range(distances.shape[0]), min_indices] <= (80 / 2)
            filtered_mask_points = mask_points[filter_condition]
            filtered_min_indices = min_indices[filter_condition]

            # Group mask points by nearest bounding box
            unique_indices, inverse_indices = torch.unique(filtered_min_indices, return_inverse=True)
            grouped_points = torch.zeros(inverse_indices.size(0), num_gts, 2, device=filtered_mask_points.device, dtype=filtered_mask_points.dtype)
            grouped_points[range(inverse_indices.size(0)), inverse_indices, :] = filtered_mask_points

            rotated_corners = [] 
            for i in range(num_gts):
                inds_now = (inverse_indices == i).nonzero().reshape(-1)
                if len(inds_now) == 0:
                    rotated_corners.append(gt_bboxes[i,:].repeat(4))
                    continue
                points_now = grouped_points[inds_now,i,:].cpu().numpy().astype(np.int32)
                rect = cv2.minAreaRect(points_now)
                box = cv2.boxPoints(rect)
                box = torch.tensor(np.int0(box), device=gt_bboxes.device, dtype=gt_bboxes.dtype).reshape(-1)
                rotated_corners.append(box)
            rotated_corners = torch.stack(rotated_corners, dim=0)
            pseudo_rbbox = poly2obb(rotated_corners, version='le90')

            scores_sum = cls_scores_average + self.w1
            pseudo_rbbox = ((self.w1/scores_sum).repeat(1,5)) * pseudo_rbbox + ((cls_scores_average/scores_sum).repeat(1,5)) * rbbox_preds_average
            pseudo_rbbox = pseudo_rbbox.clone().detach()
            # store the new information
            self.gt_pred_bboxes_bank[image_name] = pseudo_rbbox
            self.gt_pred_scores_bank[image_name] = cls_scores_average.clone().detach()
        else:
            scores_sum = cls_scores_average + self.w1
            pseudo_rbbox = ((self.w1/scores_sum).repeat(1,5)) * gt_store_bboxes + ((cls_scores_average/scores_sum).repeat(1,5)) * rbbox_preds_average
            pseudo_rbbox = pseudo_rbbox.clone().detach()

        nan_decide = torch.isnan(pseudo_rbbox).any(dim=1)
        abandoned_inds = (nan_decide == True).nonzero().reshape(-1)
        pseudo_rbbox[abandoned_inds, :2] = gt_bboxes[abandoned_inds,:]
        pseudo_rbbox[abandoned_inds, 2:] = 0

        sam_overlap = rbbox_overlaps(pseudo_rbbox, gt_bboxes_ref, mode='iou', is_aligned=True).nanmean()

        ### Generate bbox target
        areas = pseudo_rbbox[:, 2] * pseudo_rbbox[:, 3]
        areas = areas[None].expand(num_points, num_gts)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        pseudo_rbbox = pseudo_rbbox[None].expand(num_points, num_gts, 5)

        gt_ctr, gt_wh, gt_angle = torch.split(pseudo_rbbox, [2, 2, 1], dim=2)
        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(num_points, num_gts, 2, 2)
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
        inds[pos_valid_inds] = valid_gt_indices
        
        bbox_targets = bbox_targets[range(num_points), inds]
        angle_targets = gt_angle[range(num_points), inds]

        labels_pseudo = self.num_classes * torch.ones(num_points, dtype=torch.long, device=gt_bboxes.device)
        labels_pseudo[pos_valid_inds] = assign_result.labels[pos_valid_inds]

        rbbox_preds_average = rbbox_preds_average if rbbox_preds_average.numel() else torch.empty((0, 5), device=gt_labels.device)
        bbox_preds_dxdy_average = bbox_preds_dxdy_average if bbox_preds_dxdy_average.numel() else torch.empty((0, bbox_preds_dxdy_average.shape[1]), device=gt_labels.device)
        bbox_preds_theta_average = bbox_preds_theta_average if bbox_preds_theta_average.numel() else torch.empty((0, bbox_preds_theta_average.shape[1]), device=gt_labels.device)
        bbox_preds_dwdh_average = bbox_preds_dwdh_average if bbox_preds_dwdh_average.numel() else torch.empty((0, bbox_preds_theta_average.shape[1]), device=gt_labels.device)
        
        return labels, rbbox_preds_average, bbox_preds_dxdy_average, bbox_preds_theta_average, \
            bbox_preds_dwdh_average, gt_bboxes_ref, pos_gts, \
            sam_overlap, labels_pseudo, bbox_targets, angle_targets
        
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
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
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
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
        for cls_score, bbox_pred, points in zip(
                cls_scores, bbox_preds,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = bbox_pred
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_bboxes.new_full((mlvl_bboxes.shape[0],), 1))
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
    
    def concat_per_img(self, num_levels, flatten_cls_scores, flatten_bbox_preds_dxdy, flatten_bbox_preds_theta, flatten_bbox_preds_dwdh,
                        flatten_bbox_preds_cxcywhtheta, flatten_objectness):
        
        concat_cls_scores = []
        concat_bbox_preds_dxdy = []
        concat_bbox_preds_theta = []
        concat_bbox_preds_dwdh = []
        concat_bbox_preds_cxcywhtheta = []
        concat_objectness = []

        for i in range(flatten_cls_scores[0].shape[0]):
            temp_scores = []
            temp_dxdy = []
            temp_theta = []
            temp_dwdh = []
            temp_cxcywhtheta = []
            temp_objectness = []
            for j in range(num_levels):
                temp_scores.append(flatten_cls_scores[j][i,:,:])
                temp_dxdy.append(flatten_bbox_preds_dxdy[j][i,:,:])
                temp_theta.append(flatten_bbox_preds_theta[j][i,:,:])
                temp_dwdh.append(flatten_bbox_preds_dwdh[j][i,:,:])
                temp_cxcywhtheta.append(flatten_bbox_preds_cxcywhtheta[j][i,:,:])
                temp_objectness.append(flatten_objectness[j][i,:,:])
                
            concat_cls_scores.append(torch.cat(temp_scores, dim=0))
            concat_bbox_preds_dxdy.append(torch.cat(temp_dxdy, dim=0))
            concat_bbox_preds_theta.append(torch.cat(temp_theta, dim=0))
            concat_bbox_preds_dwdh.append(torch.cat(temp_dwdh, dim=0))
            concat_bbox_preds_cxcywhtheta.append(torch.cat(temp_cxcywhtheta, dim=0))
            concat_objectness.append(torch.cat(temp_objectness, dim=0))
            
        return concat_cls_scores, concat_bbox_preds_dxdy, concat_bbox_preds_theta, \
            concat_bbox_preds_dwdh, concat_bbox_preds_cxcywhtheta, concat_objectness
    
    def generate_per_img_all(self, num_imgs, cls_scores, bbox_preds_dxdy, bbox_preds_theta, bbox_preds_dwdh, bbox_preds_cxcywhtheta, 
                             objectness):
        flatten_cls_scores_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]

        flatten_bbox_preds_dxdy_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for cls_score in bbox_preds_dxdy
        ]

        flatten_bbox_preds_theta_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for cls_score in bbox_preds_theta
        ]

        flatten_bbox_preds_dwdh_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for cls_score in bbox_preds_dwdh
        ]

        flatten_bbox_preds_cxcywhtheta_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 5)
            for cls_score in bbox_preds_cxcywhtheta
        ]
        
        flatten_objectness_img = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for cls_score in objectness
        ]

        return flatten_cls_scores_img, flatten_bbox_preds_dxdy_img, flatten_bbox_preds_theta_img, flatten_bbox_preds_dwdh_img, \
               flatten_bbox_preds_cxcywhtheta_img, flatten_objectness_img
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_decode', 'bbox_preds_refine', 
                  'angle_preds', 'angle_preds_decode', 'angle_preds_refine', 'all_level_points'))
    def loss(self,
            cls_scores,
            bbox_preds,
            bbox_preds_decode,
            bbox_preds_refine,
            angle_preds,
            angle_preds_decode,
            angle_preds_refine,
            all_level_points,
            gt_points,
            gt_labels,
            gt_bboxes_ref,
            img_metas,
            insider,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores_img, flatten_bbox_preds_img, flatten_bbox_preds_decode_img, flatten_bbox_preds_refine_img, \
                            flatten_angle_preds_img, flatten_angle_preds_decode_img, flatten_angle_preds_refine_img = self.generate_per_img_all(num_imgs, cls_scores, bbox_preds, bbox_preds_decode,
                                                                                                                                                bbox_preds_refine, angle_preds, angle_preds_decode, angle_preds_refine)

        results = self.get_results(all_level_points, gt_points[0]+gt_points[1], gt_labels[0]+gt_labels[1], gt_bboxes_ref[0]+gt_bboxes_ref[1], 
                                                            insider[0]+insider[1], img_metas[0]+img_metas[1], 
                                                            flatten_cls_scores_img, flatten_bbox_preds_img, flatten_angle_preds_img,
                                                            flatten_bbox_preds_decode_img, flatten_angle_preds_decode_img,
                                                            flatten_bbox_preds_refine_img, flatten_angle_preds_refine_img)
        
        labels, rbbox_preds_list, bbox_preds_distribution_list, angle_preds_distribution_list, rbbox_preds_refine_list, gt_bboxes_insider_list = results
        
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels = torch.cat(labels)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        
        _1st_preds = torch.cat(rbbox_preds_list).clone().detach()
        _2nd_preds = torch.cat(rbbox_preds_refine_list).clone().detach()
        gt_bboxes_insider = torch.cat(gt_bboxes_insider_list)
        _1st_rbox_iou = rbbox_overlaps(_1st_preds, gt_bboxes_insider, mode='iou', is_aligned=True).mean()
        _2nd_rbox_iou = rbbox_overlaps(_2nd_preds, gt_bboxes_insider, mode='iou', is_aligned=True).mean()

        return loss_cls, _1st_rbox_iou, _2nd_rbox_iou, rbbox_preds_list, bbox_preds_distribution_list, angle_preds_distribution_list, rbbox_preds_refine_list
    
    def obb2poly_le90(self, rboxes):
        """Convert oriented bounding boxes to polygons.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        """
        B, _, H, W = rboxes.shape[0]
        N = B * H * W
        rboxes = rboxes.permute(0,2,3,1).reshape(-1, 5)
        
        x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
            1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
        tl_x, tl_y, br_x, br_y = \
            -width * 0.5, -height * 0.5, \
            width * 0.5, height * 0.5
        rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                            dim=0).reshape(2, 4, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                            N).permute(2, 0, 1)
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        polys[:, ::2] += x_ctr.unsqueeze(1)
        polys[:, 1::2] += y_ctr.unsqueeze(1)
        polys = (polys.contiguous()).reshape(B,H,W,8).permute(0,3,1,2)
        return polys
    
    def compute_mean_prob_and_indices(self, probs):
        """
        Computes the average probabilities for all possible combinations of indices in a tensor of size (N, 4, 9).
        Also returns the indices of these combinations.

        Args:
            probs (torch.Tensor): A tensor of shape (N, 4, 9) representing the probabilities.

        Returns:
            tuple: A tuple containing:
                - mean_probs (torch.Tensor): A tensor of shape (N, 9**4) with averaged probabilities.
                - combinations_tensor (torch.Tensor): A tensor of shape (9**4, 4) with the combinations of indices.
                - indices_tensor (torch.Tensor): A tensor of shape (N, 9**4, 4) with the coordinates (indices).
        """
        N, _, num_classes = probs.shape
        num_combinations = num_classes ** 4

        # Generate all index combinations
        combinations = list(itertools.product(range(num_classes), repeat=4))
        combinations_tensor = torch.tensor(combinations, device=probs.device)  # shape (9^4, 4)

        # Prepare output tensors
        mean_probs = torch.zeros((N, num_combinations), device=probs.device)  # shape (N, 9^4)
        indices_tensor = torch.zeros((N, num_combinations, 4), device=probs.device, dtype=torch.long)  # shape (N, 9^4, 4)

        for i, comb in enumerate(combinations_tensor):
            # Gather the probabilities for each combination
            comb_probs = probs[:, torch.arange(4), comb]  # shape (N, 4)
            mean_probs[:, i] = torch.mean(comb_probs, dim=1)  # shape (N)
            indices_tensor[:, i] = comb  # shape (N, 4), broadcasted to (N, 9^4, 4)

        return mean_probs, combinations_tensor, indices_tensor


    def compute_mean_prob_and_average_scores(self, probs, class_scores):
        """
        Computes the average probabilities and average class scores for all possible combinations 
        of indices in a tensor of size (N, 4, 9). Also returns the indices of these combinations.

        Args:
            probs (torch.Tensor): A tensor of shape (N, 4, 9) representing the probabilities.
            class_scores (torch.Tensor): A tensor of shape (N, 4, 9, C) representing the class scores.

        Returns:
            tuple: A tuple containing:
                - mean_probs (torch.Tensor): A tensor of shape (N, 9**4) with averaged probabilities.
                - average_class_scores (torch.Tensor): A tensor of shape (N, 9**4, C) with average class scores.
                - combinations_tensor (torch.Tensor): A tensor of shape (9**4, 4) with the combinations of indices.
                - indices_tensor (torch.Tensor): A tensor of shape (N, 9**4, 4) with the coordinates (indices).
        """
        N, num_points, num_classes = probs.shape
        _, _, _, C = class_scores.shape
        num_combinations = num_classes ** 4

        # Generate all index combinations
        combinations_tensor = torch.tensor(list(itertools.product(range(num_classes), repeat=4)), device=probs.device)  # shape (9^4, 4)

        # Prepare output tensors
        mean_probs = torch.zeros((N, num_combinations), device=probs.device)  # shape (N, 9^4)
        average_class_scores = torch.zeros((N, num_combinations, C), device=probs.device)  # shape (N, 9^4, C)

        # Generate indices for advanced indexing
        comb_indices = combinations_tensor.unsqueeze(0).repeat(N, 1, 1)  # shape (N, 9^4, 4)
        batch_indices = torch.arange(N, device=probs.device).view(-1, 1, 1).expand(N, num_combinations, num_points)  # shape (N, 9^4, 4)
        point_indices = torch.arange(num_points, device=probs.device).view(1, 1, -1).expand(N, num_combinations, num_points)  # shape (N, 9^4, 4)

        # Reshape for gathering
        gather_indices = comb_indices.permute(1, 2, 0).reshape(num_combinations, -1)  # shape (9^4, 4N)

        # Gather the probabilities and class scores for each combination
        probs_gathered = probs[batch_indices.reshape(-1), point_indices.reshape(-1), gather_indices.reshape(-1)].reshape(num_combinations, N, num_points)  # shape (9^4, N, 4)
        class_scores_gathered = class_scores[batch_indices.reshape(-1), point_indices.reshape(-1), gather_indices.reshape(-1), :].reshape(num_combinations, N, num_points, C)  # shape (9^4, N, 4, C)

        # Compute mean probabilities and average class scores
        # mean_probs = probs_gathered.mean(dim=2).t()  # shape (N, 9^4)
        # average_class_scores = class_scores_gathered.mean(dim=2).permute(1, 0, 2)  # shape (N, 9^4, C)
        mean_probs = probs_gathered.prod(dim=2).t()  # shape (N, 9^4)
        average_class_scores = class_scores_gathered.prod(dim=2).permute(1, 0, 2)  # shape (N, 9^4, C)

        indices_tensor = comb_indices

        return mean_probs, average_class_scores, combinations_tensor, indices_tensor
