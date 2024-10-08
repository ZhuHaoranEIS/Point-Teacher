# Copyright (c) OpenMMLab. All rights reserved.
import torch
from scipy.optimize import linear_sum_assignment

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from ..iou_calculators import build_iou_calculator


@BBOX_ASSIGNERS.register_module()
class FUSETopkAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 num_pre=7,
                 topk=5,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='PointCost', mode='L1', weight=1.0),
                 location_cost=dict(type='InsiderCost', weight=2.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.location_cost = build_match_cost(location_cost)
        self.topk = topk
        self.num_pre = num_pre
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))

    def assign(self,
               bbox_pred,
               points,
               cls_pred,
               centerness,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_bboxes = bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if gt_bboxes == None:
            # No ground truth or boxes, return empty assignment
            assigned_gt_inds[:] = 0
            return AssignResult(
                0, assigned_gt_inds, None, labels=assigned_labels)
        
        num_gts = gt_bboxes.size(0)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        
        # 2. compute the weighted costs
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        location_cost = self.location_cost(bbox_pred, gt_bboxes)
        reg_cost = self.reg_cost(points, gt_bboxes)
        
        # first assign
        cost = reg_cost.detach()
        topk_values, topk_indices = torch.topk(cost, self.num_pre, dim=0, largest=False)
        column_indices = torch.arange(cost.size(1)).unsqueeze(0).repeat(self.num_pre, 1)
        matched_row_inds_pre = topk_indices.flatten().to(assigned_gt_inds.device)
        matched_col_inds_pre = column_indices.flatten().to(assigned_gt_inds.device)
        
        # second assign
        cost = (cls_cost + location_cost).detach()
        assigned_gt_inds[:] = 0
        for i in range(num_gts):
            inds = (matched_col_inds_pre == i).nonzero().reshape(-1)
            if inds.numel() == 0:
                continue # filter
            row_inds = matched_row_inds_pre[inds]
            cost_i = cost[row_inds,:]
            if inds.numel() <= self.topk:
                assigned_gt_inds[row_inds] = i + 1
                assigned_labels[row_inds] = gt_labels[i]
                continue
            topk_values, topk_indices = torch.topk(cost_i, self.topk, dim=0, largest=False)
            matched_row_inds_i = topk_indices.flatten().to(assigned_gt_inds.device)
            assigned_gt_inds[row_inds[matched_row_inds_i]] = i + 1
            assigned_labels[row_inds[matched_row_inds_i]] = gt_labels[i]

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
        