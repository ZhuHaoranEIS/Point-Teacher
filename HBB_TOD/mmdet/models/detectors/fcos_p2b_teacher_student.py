from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_detector
from .single_stage import SingleStageDetector
from .base import BaseDetector
from mmdet.core import bbox2result

import torch
import torch.nn as nn
import numpy as np
import cv2

from mmdet.core import multi_apply, reduce_mean

from .syn_images_generator_v2 import load_basic_shape, generate_black_paper, strong_augmentation, gen_negative_proposals, \
                                     fine_proposals_from_cfg, gen_proposals_from_cfg, MIL_gen_proposals_from_cfg

from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, distance2bbox, bbox2distance, 
                        build_assigner, build_sampler, bbox_overlaps)
from mmdet.core.bbox.iou_calculators import bbox_overlaps

import torch.nn.functional as F
import math
import time
import mmcv

import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import os
from .data_augument_bank import random_point_in_quadrilateral, obb2poly_le90



@DETECTORS.register_module()
class TS_P2B_FCOS(BaseDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 _model_,
                 _point_='random',
                 num_stages=2,
                 num_refine=500,
                 num_training_burninstep1=512,
                 num_training_burninstep2=512,
                 ema_alpha=0.999,
                 filter_score=0.8,
                 burn_in_step=10000,
                 lamda=1.0,
                 alpha=[0.1, 1.0],
                 shape_list = [[20, 20, 0.5, 0.5], [30, 120, 0.5, 0.5], [10, 20, 0.5, 0.5],
                               [20, 50, 0.5, 0.5], [30, 20, 0.5, 0.5]],
                 MIL_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TS_P2B_FCOS, self).__init__(init_cfg)
        self.teacher = build_detector(_model_, train_cfg, test_cfg)
        self.student = build_detector(_model_, train_cfg, test_cfg)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ### keep the fixed single point
        self.gt_bboxes_point = {}
        self.refined_gt_bboxes_point = {}
        ### augumentation type count
        self.count = 0
        self.ema_alpha = ema_alpha
        self.epoch = 0
        self.epoch_dict = {}
        self.max_epoch = 12
        self.lamda = lamda
        ### synthetic
        self.pattern, self.prior_size = load_basic_shape(shape_list)
        self.scale_ratio = 1.0
        self.filter_score = filter_score
        self.burn_in_step = burn_in_step
        self.alpha = alpha
        
        ### MIL
        self.num_stages = num_stages
        self.num_refine = num_refine
        self.num_training_burninstep1 = num_training_burninstep1
        self.num_training_burninstep2 = num_training_burninstep2

        self._point_ = _point_
        if self.train_cfg != None:
            self.fine_proposal_cfg = []
            self.fine_proposal_extensive_cfg = []
            for i in range(len(self.train_cfg['fine_proposal_cfg'])):
                self.fine_proposal_cfg.append(self.train_cfg['fine_proposal_cfg'][i])
                self.fine_proposal_extensive_cfg.append(self.train_cfg['fine_proposal_extensive_cfg'][i])
    
    def extract_feat(self, img, model=None):
        """Directly extract features from the backbone+neck."""
        x = model.extract_feat(img)
        return x

    def forward_dummy(self, img, model=None):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = model.forward_dummy(img)
        return outs
    
    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        '''initalization'''
        img = img.to(torch.float); device = gt_bboxes[0].device; num_img = len(img_metas)
        B, C, H, W = img.shape
        '''update teacher parameters'''
        self.update_teacher_model(self.teacher, self.student, self.ema_alpha)
        '''update epoch'''
        self.update_epoch(num_img, img_metas)
        '''generate point labels'''
        gt_points, img_list, img = self.genrate_points(num_img, img, img_metas, gt_bboxes)

        losses = {}
        if self.count <= self.burn_in_step:
            losses = self.forward_train_burn_in_step1(num_img, img, img_list, img_metas, gt_bboxes, gt_points, gt_labels, gt_bboxes_ignore, device)
        else:
            losses = self.forward_train_burn_in_step2(num_img, img, img_list, img_metas, gt_bboxes, gt_points, gt_labels, gt_bboxes_ignore, device)

        self.count += 1  
        return losses
    
    def forward_train_burn_in_step1(self, num_img, img, img_list, img_metas, gt_bboxes, gt_points, gt_labels, gt_bboxes_ignore, device):
        losses = {}
        img_syn, img_syn_list, gt_bboxes_synthetic = self.genrate_syn(num_img, img_list, gt_bboxes, gt_labels)

        # PALETTENOISE = [(0, 255, 0), (0,  0, 255), (255, 144, 30), (0, 173, 205), (128, 128, 0)]
        # for i in range(num_img):
        #     image_path = img_metas[i]['ori_filename']
        #     image_store_path = "/home/zhr/mmdet-rfla/mmdet/models/detectors/syn"
        #     image_store_path = os.path.join(image_store_path, image_path+'_'+str(self.count)+'.png')
        #     img_plot = img_syn[i,:,:,:].permute(1, 2, 0).contiguous().cpu().numpy()
        #     cv2.imwrite(image_store_path, img_plot)
        #     image = cv2.imread(image_store_path)
        #     for j in range(gt_bboxes_synthetic[i].shape[0]):
        #         x1, y1, x2, y2 =  gt_bboxes_synthetic[i][j,:].tolist()
        #         color = PALETTENOISE[0]
        #         image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        #     cv2.imwrite(image_store_path, image) 

        img_all = torch.cat([img_syn, img], dim=0)
        student_feature_all = self.extract_feat(img_all, self.student)
        mil_feature_student_all = self.student.bbox_head.forward_mil(student_feature_all)
        student_feature_syn, mil_feature_student_syn, mil_feature_student_ori = [], [], []
        for i in range(len(student_feature_all)):
            student_feature_syn.append(student_feature_all[i][:num_img,:,:,:])
            mil_feature_student_syn.append(mil_feature_student_all[i][:num_img,:,:,:])
            mil_feature_student_ori.append(mil_feature_student_all[i][num_img:,:,:,:])
        del student_feature_all; del mil_feature_student_all; del img_all; torch.cuda.empty_cache()

        student_detection_results_syn_img = self.student.bbox_head(student_feature_syn)
        loss_student_syn = self.student.bbox_head.loss(*student_detection_results_syn_img, gt_bboxes_synthetic, img_metas, gt_bboxes_ignore)
        del img_syn; del img_syn_list; del student_detection_results_syn_img; del student_feature_syn; torch.cuda.empty_cache()

        with torch.no_grad():
            teacher_feature_img = self.extract_feat(img, self.teacher)
            teacher_detection_results_ori_img = self.teacher.bbox_head(teacher_feature_img)
            pseudo_bboxes_coarse, pseudo_points_coarse, pseudo_labels_coarse, _, _ = self.teacher.bbox_head.get_pseudo_bbox(*teacher_detection_results_ori_img, 
                                                                                                                                gt_points, gt_labels, gt_bboxes,
                                                                                                                                self.filter_score, img_metas, 
                                                                                                                                img_list, gt_bboxes_ignore)
            del teacher_feature_img; del teacher_detection_results_ori_img; torch.cuda.empty_cache()

        pseudo_bboxes_refine, pseudo_points_refine, mil_losses = self.forward_mil_head_burn_in_step1(num_img, gt_bboxes_synthetic, pseudo_bboxes_coarse, 
                                                                                                     pseudo_points_coarse, pseudo_labels_coarse, gt_bboxes, 
                                                                                                     img_metas, mil_feature_student_syn, mil_feature_student_ori,
                                                                                                     img)
        
        pseudo_bboxes_refine, pseudo_points_refine = pseudo_bboxes_coarse, pseudo_points_coarse

        if mil_losses != None:
            losses.update(mil_losses)
            gt_points = self.update_points(num_img, img_metas, pseudo_bboxes_refine)
            real_point = bbox_xyxy_to_cxcywh(torch.cat(gt_bboxes,dim=0))
            losses['refined_points_distance'] = (torch.sqrt((torch.cat(gt_points) - real_point[:,:2]) ** 2) / torch.sqrt((real_point[:,2:] / 2) ** 2)).mean()
        del mil_feature_student_syn; del mil_feature_student_ori; del mil_losses; torch.cuda.empty_cache()
        
        img_aug, img_aug_list, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine = strong_augmentation(
                                                                    img, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_coarse, pseudo_bboxes_refine)

        student_feature_aug = self.extract_feat(img_aug, self.student)
        student_detection_results_aug_img = self.student.bbox_head(student_feature_aug)
        del student_feature_aug
        loss_student_pseudo = self.student.bbox_head.loss_pseudo(*student_detection_results_aug_img, gt_points, gt_labels, 
                                                                 pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine, 
                                                                 [None for i in range(num_img)], img_metas, img_aug_list, self.count<=self.burn_in_step, gt_bboxes_ignore)

        losses['loss_cls'] = loss_student_pseudo[0]
        losses['loss_bbox'] = loss_student_syn[0]
        losses['loss_centerness'] = loss_student_syn[1]
        del loss_student_pseudo; del student_detection_results_aug_img; del loss_student_syn; torch.cuda.empty_cache()

        return losses

    def forward_train_burn_in_step2(self, num_img, img, img_list, img_metas, gt_bboxes, gt_points, gt_labels, gt_bboxes_ignore, device):
        losses = {}
        with torch.no_grad():
            teacher_feature_img = self.extract_feat(img, self.teacher)
            teacher_detection_results_ori_img = self.teacher.bbox_head(teacher_feature_img)
            pseudo_bboxes_coarse, pseudo_points_coarse, pseudo_labels_coarse, _, _ = self.teacher.bbox_head.get_pseudo_bbox(*teacher_detection_results_ori_img, 
                                                                                                                                gt_points, gt_labels, gt_bboxes,
                                                                                                                                self.filter_score, img_metas, 
                                                                                                                                img_list, gt_bboxes_ignore)
            del teacher_feature_img; del teacher_detection_results_ori_img; torch.cuda.empty_cache()
        student_feature_img = self.extract_feat(img, self.student)
        mil_feature_student = self.student.bbox_head.forward_mil(student_feature_img)
        pseudo_bboxes_refine, pseudo_points_refine, mil_losses = self.forward_mil_head_burn_in_step2(num_img, pseudo_bboxes_coarse, pseudo_points_coarse, 
                                                                                                     pseudo_labels_coarse, gt_bboxes, img_metas, 
                                                                                                     mil_feature_student)
        
        if mil_losses != None:
            losses.update(mil_losses)
        del student_feature_img; del mil_feature_student; del mil_losses; torch.cuda.empty_cache()

        gt_points = self.update_points(num_img, img_metas, pseudo_bboxes_refine)
        real_point = bbox_xyxy_to_cxcywh(torch.cat(gt_bboxes,dim=0))
        losses['refined_points_distance'] = (torch.sqrt((torch.cat(gt_points) - real_point[:,:2]) ** 2) / torch.sqrt((real_point[:,2:] / 2) ** 2)).mean()
        
        img_aug, img_aug_list, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine = strong_augmentation(
                                                                    img, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_coarse, pseudo_bboxes_refine)
        

        student_feature_aug = self.extract_feat(img_aug, self.student)
        student_detection_results_aug_img = self.student.bbox_head(student_feature_aug)
        del student_feature_aug
        loss_student_pseudo = self.student.bbox_head.loss_pseudo(*student_detection_results_aug_img, gt_points, gt_labels, 
                                                                 pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine, 
                                                                 [None for i in range(num_img)], img_metas, img_aug_list, self.count<=self.burn_in_step, gt_bboxes_ignore)

        losses['loss_cls'] = loss_student_pseudo[0]
        losses['loss_bbox'] = loss_student_pseudo[1]
        losses['loss_centerness'] = loss_student_pseudo[2]
        del loss_student_pseudo; del student_detection_results_aug_img; torch.cuda.empty_cache()
        return losses
    
    def update_teacher_model(self, teacher_model, student_model, ema_decay=0.999):
        with torch.no_grad():
            for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def update_epoch(self, num_img, img_metas):
        if img_metas[0]['ori_filename'] in self.epoch_dict.keys():
            self.epoch += 1
            self.epoch_dict = {}
        for i in range(num_img):
            self.epoch_dict[img_metas[i]['ori_filename']] = 1

    def update_points(self, num_img, img_metas, pseudo_bboxes):
        refined_gt_points = []
        for i in range(num_img):
            pseudo_center = bbox_xyxy_to_cxcywh(pseudo_bboxes[i])[:,:2]
            origin_center = self.gt_bboxes_point[img_metas[i]['ori_filename']]
            refine_center = (1-self.lamda) * pseudo_center + (self.lamda) * origin_center
            refined_gt_points.append(refine_center)
            self.refined_gt_bboxes_point[img_metas[i]['ori_filename']] = refine_center
        return refined_gt_points
            
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img = img.to(torch.float)
        feat = self.extract_feat(img, self.teacher)
        results_list = self.teacher.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.teacher.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        img = img.to(torch.float)
        feats = self.extract_feats(imgs)
        results_list = self.teacher.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
    
    def filter_pseudo_bbox(self, num_img, pseudo_bboxes, pseudo_labels, valid_inds):
        valid = [(valid_inds[i] != 0).nonzero().reshape(-1) for i in range(num_img)]
        pseudo_bboxes_new = [pseudo_bboxes[i][valid[i],:] for i in range(num_img)]
        pseudo_points_new = [bbox_xyxy_to_cxcywh(pseudo_bboxes_new[i])[:,:2] for i in range(num_img)]
        pseudo_labels_new = [pseudo_labels[i][valid[i]] for i in range(num_img)]
        return pseudo_bboxes_new, pseudo_points_new, pseudo_labels_new
    
    def forward_mil_head_burn_in_step1(self, num_img, synthetic_bboxes, pseudo_bboxes, pseudo_points, pseudo_labels, gt_bboxes, img_metas, 
                                       x_synthetic, x_ori, img):
        losses = {}
        syn_valid_inds = [i for i in range(num_img) if synthetic_bboxes[i].shape[0] != 0]
        if len(syn_valid_inds) != num_img:  
            return [torch.empty(0,4,device=pseudo_bboxes[0].device,dtype=pseudo_bboxes[0].dtype) for i in range(num_img)], \
                   [torch.empty(0,2,device=pseudo_bboxes[0].device,dtype=pseudo_bboxes[0].dtype) for i in range(num_img)], \
                   None
        synthetic_bboxes_training, synthetic_points_training, pseudo_bboxes_training, gt_bboxes_training, pseudo_labels_training, pseudo_points_training = [], [], [], [], [], []
        refined_pseudo_bboxes = []; refined_pseudo_points = []
        for i in range(num_img):
            synthetic_bboxes_training.append(synthetic_bboxes[i][:self.num_training_burninstep1,:])
            synthetic_points_training.append(bbox_xyxy_to_cxcywh(synthetic_bboxes_training[-1])[:,:2])
            pseudo_bboxes_training.append(pseudo_bboxes[i][:self.num_training_burninstep1,:])
            gt_bboxes_training.append(gt_bboxes[i][:self.num_training_burninstep1,:])
            pseudo_points_training.append(pseudo_points[i][:self.num_training_burninstep1,:])
            pseudo_labels_training.append(pseudo_labels[i][:self.num_training_burninstep1])
            refined_pseudo_bboxes.append(pseudo_bboxes[i].clone())
            refined_pseudo_points.append(pseudo_points[i].clone())
        coarse_bboxes_iou = bbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                          torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
        losses['coarse_bboxes_iou'] = coarse_bboxes_iou
        for stage in range(self.num_stages):
            proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list = MIL_gen_proposals_from_cfg(pseudo_points_training, pseudo_bboxes_training, 
                                                                                                                             self.fine_proposal_cfg[stage], 
                                                                                                                             gt_bboxes_training, img_meta=img_metas)
            syn_proposals_list, syn_proposals_valid_list, syn_proposals_reference_list, syn_proposals_real_list = MIL_gen_proposals_from_cfg(synthetic_points_training, synthetic_bboxes_training, 
                                                                                                                                             self.fine_proposal_cfg[stage],
                                                                                                                                             synthetic_bboxes_training, img_meta=img_metas)
            neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_points_training, self.fine_proposal_cfg[stage],
                                                                        proposals_list, img_meta=img_metas)
            mil_loss, pseudo_bboxes_training = self.student.bbox_head.MIL_head_burn_in_step1(x_ori,
                                                                                             x_synthetic,
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
                                                                                             synthetic_bboxes_training,
                                                                                             pseudo_bboxes_training,
                                                                                             pseudo_labels_training,
                                                                                             self.fine_proposal_extensive_cfg[stage],
                                                                                             stage)
            refined_bboxes_iou = bbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                               torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
            losses[f'stage{stage}_refine_bboxes_iou'] = refined_bboxes_iou
            mil_loss[f'stage{stage}_loss_mil_bbox'] = mil_loss[f'stage{stage}_loss_mil_bbox'] * self.alpha[0]
            mil_loss[f'stage{stage}_loss_mil_bags'] = mil_loss[f'stage{stage}_loss_mil_bags'] * self.alpha[1]
            losses.update(mil_loss)
        for i in range(num_img):
            refined_pseudo_bboxes[i][:self.num_training_burninstep1,:] = pseudo_bboxes_training[i]
            refined_pseudo_points[i][:self.num_training_burninstep1,:] = bbox_xyxy_to_cxcywh(pseudo_bboxes_training[i])[:,:2]
        return refined_pseudo_bboxes, refined_pseudo_points, losses

    def forward_mil_head_burn_in_step2(self, num_img, pseudo_bboxes, pseudo_points, pseudo_labels, gt_bboxes, img_metas, x_ori):
        losses = {}
        pseudo_bboxes_training, gt_bboxes_training, pseudo_labels_training, pseudo_points_training = [], [], [], []
        refined_pseudo_bboxes = []; refined_pseudo_points = []
        for i in range(num_img):
            pseudo_bboxes_training.append(pseudo_bboxes[i][:self.num_training_burninstep2,:].clone())
            gt_bboxes_training.append(gt_bboxes[i][:self.num_training_burninstep2,:].clone())
            pseudo_points_training.append(pseudo_points[i][:self.num_training_burninstep2,:].clone())
            pseudo_labels_training.append(pseudo_labels[i][:self.num_training_burninstep2].clone())
            refined_pseudo_bboxes.append(pseudo_bboxes[i].clone())
            refined_pseudo_points.append(pseudo_points[i].clone())
        coarse_bboxes_iou = bbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                            torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
        losses['coarse_bboxes_iou'] = coarse_bboxes_iou
        for stage in range(self.num_stages):
            proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list = MIL_gen_proposals_from_cfg(pseudo_points_training, pseudo_bboxes_training, 
                                                                                                                             self.fine_proposal_cfg[stage], 
                                                                                                                             gt_bboxes_training, img_meta=img_metas)
            neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_points_training, self.fine_proposal_cfg[stage],
                                                                        proposals_list, img_meta=img_metas)
            mil_loss, pseudo_bboxes_training = self.student.bbox_head.MIL_head_burn_in_step2(x_ori,
                                                                                             img_metas,
                                                                                             proposals_list,
                                                                                             proposals_valid_list,
                                                                                             proposals_reference_list,
                                                                                             proposals_real_list,
                                                                                             neg_proposal_list,
                                                                                             neg_weight_list,
                                                                                             pseudo_bboxes_training,
                                                                                             pseudo_labels_training,
                                                                                             self.fine_proposal_extensive_cfg[stage],
                                                                                             stage)
            refined_bboxes_iou = bbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                               torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
            losses[f'stage{stage}_refine_bboxes_iou'] = refined_bboxes_iou
            mil_loss[f'stage{stage}_loss_mil_bbox'] = mil_loss[f'stage{stage}_loss_mil_bbox'] * self.alpha[0]
            mil_loss[f'stage{stage}_loss_mil_bags'] = mil_loss[f'stage{stage}_loss_mil_bags'] * self.alpha[1]
            losses.update(mil_loss)
        for i in range(num_img):
            refined_pseudo_bboxes[i][:self.num_training_burninstep2,:] = pseudo_bboxes_training[i]
            refined_pseudo_points[i][:self.num_training_burninstep2,:] = bbox_xyxy_to_cxcywh(pseudo_bboxes_training[i])[:,:2]
        return refined_pseudo_bboxes, refined_pseudo_points, losses
    

    def genrate_syn(self, num_img, img_list, gt_bboxes, gt_labels):
        def synthesis_single(img, bboxes, labels, device='cpu'):
            values = torch.tensor(range(len(self.pattern))).to(device).long()
            N = len(labels)
            indices = torch.randint(0, len(values), (N,))
            labels = values[indices]
            labels = labels[:, None]
            bboxes = bbox_xyxy_to_cxcywh(bboxes)
            bb = torch.cat((bboxes, torch.zeros_like(labels), 
                            torch.ones_like(labels), labels), -1)
            C, H, W = img.shape
            # white_paper = torch.ones_like(img.cpu()) * 255
            img, bb = generate_black_paper(
                img.cpu(), bb.cpu(), img.cpu(), self.pattern, self.prior_size, 
                range(int(len(self.pattern)/2)), min(H, W))
            img = img.to(device)
            bb = bb.to(device)
            syn_bboxes_obb = bb[:,:5]
            syn_bboxes_poly = obb2poly_le90(syn_bboxes_obb)
            min_x = syn_bboxes_poly[:,[0,2,4,6]].min(dim=1, keepdim=True)[0]
            max_x = syn_bboxes_poly[:,[0,2,4,6]].max(dim=1, keepdim=True)[0]
            min_y = syn_bboxes_poly[:,[1,3,5,7]].min(dim=1, keepdim=True)[0]
            max_y = syn_bboxes_poly[:,[1,3,5,7]].max(dim=1, keepdim=True)[0]
            syn_bboxes_xyxy = torch.cat([min_x, min_y, max_x, max_y], dim=1)
            return img, syn_bboxes_xyxy

        device = img_list[0].device
        img_sys, bb_sys = multi_apply(synthesis_single,
                                      img_list,
                                      gt_bboxes,
                                      gt_labels,
                                      device=device)
        img_sys_fuse = torch.stack(img_sys, dim=0)
        return img_sys_fuse, img_sys, bb_sys
    
    def genrate_points(self, num_img, img, img_metas, gt_bboxes):
        gt_points = []
        img_list = []
        for i in range(num_img):
            if img_metas[i]['ori_filename'] in self.refined_gt_bboxes_point.keys():
                gt_points.append(self.refined_gt_bboxes_point[img_metas[i]['ori_filename']])
            else:
                random_points = random_point_in_quadrilateral(gt_bboxes[i], self._point_)
                # if self._point_ == 'random':
                #     random_points = random_point_in_quadrilateral(gt_bboxes[i])
                # elif self._point_ == 'center':
                #     random_points = bbox_xyxy_to_cxcywh(gt_bboxes[i])[:,:2]
                gt_points.append(random_points)
                self.gt_bboxes_point[img_metas[i]['ori_filename']] = random_points
            img_list.append(img[i,:,:,:])
        return gt_points, img_list, img
    