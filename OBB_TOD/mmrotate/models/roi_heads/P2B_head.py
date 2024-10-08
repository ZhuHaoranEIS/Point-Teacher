import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
import math
import copy


from mmrotate.core import rbbox2roi
from mmrotate.core import (build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, rbbox2roi)
from ..builder import build_roi_extractor


@HEADS.register_module()
class P2BHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, bbox_roi_extractor, num_stages, bbox_head, top_k=7, with_atten=None, **kwargs):
        super(P2BHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk1 = top_k  # 7
        self.topk2 = top_k  # 7

        self.featmap_stride = bbox_roi_extractor.featmap_strides
        self.with_atten = with_atten


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        mask_head = None

    def forward_dummy(self, x, proposals):
        return None

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

        loss_instance_mil = self.bbox_head.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
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

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)
        
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


    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    stage,
                    x,
                    proposal_list,
                    proposals_valid_list,
                    gt_bboxes,
                    gt_labels,
                    gt_anns_id,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, pseudo_bboxes = self.simple_test_bboxes(
            x, img_metas, proposal_list, proposals_valid_list, gt_bboxes, gt_labels, gt_anns_id, stage, self.test_cfg,
            rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # pseudo_bboxes = [i[:, :4] for i in det_bboxes]
        return bbox_results, pseudo_bboxes

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels

