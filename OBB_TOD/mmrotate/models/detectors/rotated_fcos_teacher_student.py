# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv import ConfigDict
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from .base import BaseDetector
from ..builder import ROTATED_DETECTORS, build_detector

# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector

from mmrotate.core import rbbox2result, build_bbox_coder
from ...core.bbox.transforms import obb2poly, poly2obb
import torch
import numpy as np
import cv2

from mmdet.core import multi_apply
from .data_augument_bank import random_point_in_quadrilateral, imshow_det_rbboxes

from .syn_images_generator_v2 import load_basic_pattern, generate_sythesis, load_basic_shape, generate_black_paper, \
                                     MIL_gen_proposals_from_cfg, gen_negative_proposals, fine_proposals_from_cfg, gen_proposals_from_cfg, strong_augmentation

from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
import torchvision.transforms.functional as TF

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, distance2bbox, bbox2distance, 
                        build_assigner, build_sampler, bbox_overlaps)

from ...core.bbox.iou_calculators import rbbox_overlaps

import torch.nn.functional as F
import math
import time

import random
import torchvision.transforms as T
# import torchvision.transforms.functional as F

from mmrotate.core import (build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, rbbox2roi)

import os

@ROTATED_DETECTORS.register_module()
class RotatedFCOS_TS(RotatedSingleStageDetector):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def __init__(self,
                 _model_,
                 angle_version,
                 _point_='random',
                 num_stages=2,
                 num_refine=500,
                 num_training_burninstep1=512,
                 num_training_burninstep2=512,
                 ema_alpha=0.999,
                 filter_score=0.8,
                 burn_in_step=20000,
                 lamda=1.0,
                 alpha=[0.1, 1.0, 0.1],
                 shape_list = [[40, 40, 0.5, 0.5], [20, 40, 0.5, 0.5], [30, 120, 0.5, 0.5],
                               [40, 100, 0.5, 0.5], [60, 40, 0.5, 0.5]],
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.teacher = build_detector(_model_, train_cfg, test_cfg)
        self.student = build_detector(_model_, train_cfg, test_cfg)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.angle_version = angle_version

        ### keep the fixed single point
        self.gt_bboxes_point = {}
        self.refined_gt_bboxes_point = {}
        self.lamda = lamda
        ### augumentation type count
        self.count = 0
        self.ema_alpha = ema_alpha
        self.epoch = 0
        self.epoch_dict = {}
        self.max_epoch = 12
        ### synthetic
        # shape_list = [[40, 40, 0.5, 0.5], [30, 120, 0.5, 0.5], [20, 40, 0.5, 0.5],
        #               [40, 100, 0.5, 0.5], [60, 40, 0.5, 0.5]]
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
        device = gt_bboxes[0].device; num_img = len(img_metas)
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

        # for i in range(num_img):
        #     image_path = img_metas[i]['ori_filename']
        #     image_store_path = "/home/zhr/SODA/mmrotate/models/detectors/syn"
        #     image_store_path = os.path.join(image_store_path, image_path+'_'+str(self.count)+'.png')
        #     img_plot = img_syn[i,:,:,:].permute(1, 2, 0).contiguous().cpu().numpy()
        #     cv2.imwrite(image_store_path, img_plot)
        #     imshow_det_rbboxes(image_store_path,
        #             bboxes=gt_bboxes_synthetic[i].cpu().numpy(),
        #             labels=np.ones(gt_bboxes_synthetic[i].shape[0]).astype(int),
        #             bbox_color='green',
        #             text_color='green',
        #             out_file=image_store_path)
        
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
                                                                                                     img_metas, mil_feature_student_syn, mil_feature_student_ori)

        # for i in range(num_img):
        #     image_path = img_metas[i]['ori_filename']
        #     image_store_path = "/home/zhr/SODA/mmrotate/models/detectors/refinement"
        #     image_store_path = os.path.join(image_store_path, image_path+'_'+str(self.count)+'.png')
        #     img_plot = img[i,:,:,:].permute(1, 2, 0).contiguous().cpu().numpy()
        #     cv2.imwrite(image_store_path, img_plot)
        #     imshow_det_rbboxes(image_store_path,
        #             bboxes=pseudo_bboxes_coarse[i].cpu().numpy(),
        #             labels=np.ones(pseudo_bboxes_coarse[i].shape[0]).astype(int),
        #             bbox_color='green',
        #             text_color='green',
        #             out_file=image_store_path)
        #     imshow_det_rbboxes(image_store_path,
        #             bboxes=pseudo_bboxes_refine[i].cpu().numpy(),
        #             labels=np.ones(pseudo_bboxes_refine[i].shape[0]).astype(int),
        #             bbox_color='red',
        #             text_color='red',
        #             out_file=image_store_path)

        pseudo_bboxes_refine, pseudo_points_refine = pseudo_bboxes_coarse, pseudo_points_coarse

        if mil_losses != None:
            losses.update(mil_losses)
        gt_points = self.update_points(num_img, img_metas, pseudo_bboxes_refine)
        real_point = torch.cat(gt_bboxes,dim=0)
        losses['refined_points_distance'] = (torch.sqrt((torch.cat(gt_points) - real_point[:,:2]) ** 2) / torch.sqrt((real_point[:,2:4] / 2) ** 2)).mean()
        del mil_feature_student_syn; del mil_feature_student_ori; del mil_losses; torch.cuda.empty_cache()
        
        img_aug, img_aug_list, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine = strong_augmentation(
                                                                    img, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_coarse, pseudo_bboxes_refine, self.angle_version)
        
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
        
        # for i in range(num_img):
        #     image_path = img_metas[i]['ori_filename']
        #     image_store_path = "/home/zhr/SODA/mmrotate/models/detectors/refinement"
        #     image_store_path = os.path.join(image_store_path, image_path+'_'+str(self.count)+'.png')
        #     img_plot = img[i,:,:,:].permute(1, 2, 0).contiguous().cpu().numpy()
        #     cv2.imwrite(image_store_path, img_plot)
        #     imshow_det_rbboxes(image_store_path,
        #             bboxes=pseudo_bboxes_coarse[i].cpu().numpy(),
        #             labels=np.ones(pseudo_bboxes_coarse[i].shape[0]).astype(int),
        #             bbox_color='green',
        #             text_color='green',
        #             out_file=image_store_path)
        #     imshow_det_rbboxes(image_store_path,
        #             bboxes=pseudo_bboxes_refine[i].cpu().numpy(),
        #             labels=np.ones(pseudo_bboxes_refine[i].shape[0]).astype(int),
        #             bbox_color='red',
        #             text_color='red',
        #             out_file=image_store_path)

        if mil_losses != None:
            losses.update(mil_losses)
        gt_points = self.update_points(num_img, img_metas, pseudo_bboxes_refine)
        real_point = torch.cat(gt_bboxes, dim=0)
        losses['refined_points_distance'] = (torch.sqrt((torch.cat(gt_points) - real_point[:,:2]) ** 2) / torch.sqrt((real_point[:,2:4] / 2) ** 2)).mean()
        del student_feature_img; del mil_feature_student; del mil_losses; torch.cuda.empty_cache()
        
        img_aug, img_aug_list, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_refine, pseudo_bboxes_refine = strong_augmentation(
                                                                    img, gt_points, gt_labels, pseudo_points_refine, pseudo_labels_coarse, pseudo_bboxes_refine, self.angle_version)
        
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

    def update_points(self, num_img, img_metas, pseudo_bboxes):
        refined_gt_points = []
        for i in range(num_img):
            pseudo_center = pseudo_bboxes[i][:,:2]
            origin_center = self.gt_bboxes_point[img_metas[i]['ori_filename']]
            refine_center = (1-self.lamda) * pseudo_center + (self.lamda) * origin_center
            refined_gt_points.append(refine_center)
            self.refined_gt_bboxes_point[img_metas[i]['ori_filename']] = refine_center
        return refined_gt_points
    
    def update_teacher_model(self, teacher_model, student_model, ema_decay=0.999):
        with torch.no_grad():
            for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def update_epoch(self, num_img, img_metas):
        if img_metas[0]['filename'] in self.epoch_dict.keys():
            self.epoch += 1
            self.epoch_dict = {}
        for i in range(num_img):
            self.epoch_dict[img_metas[i]['filename']] = 1
        
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img, self.teacher)
        outs = self.teacher.bbox_head(x)
        bbox_list = self.teacher.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.teacher.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
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
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.teacher.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
    def filter_pseudo_bbox(self, num_img, pseudo_bboxes, pseudo_points, pseudo_labels, valid_inds):
        valid = [(valid_inds[i] != 0).nonzero().reshape(-1) for i in range(num_img)]
        pseudo_bboxes = [pseudo_bboxes[i][valid[i],:] for i in range(num_img)]
        pseudo_points = [pseudo_points[i][valid[i],:] for i in range(num_img)]
        pseudo_labels = [pseudo_labels[i][valid[i]] for i in range(num_img)]
        return pseudo_bboxes, pseudo_points, pseudo_labels
    
    def genrate_syn(self, num_img, img_list, gt_bboxes, gt_labels):
        def synthesis_single(img, bboxes, labels, device='cpu'):
            values = torch.tensor(range(len(self.pattern))).to(device).long()
            N = len(labels)
            indices = torch.randint(0, len(values), (N,))
            labels = values[indices]
            labels = labels[:, None]
            bb = torch.cat((bboxes, torch.ones_like(labels), labels), -1)
            C, H, W = img.shape
            img, bb = generate_black_paper(
                img.cpu(), bb.cpu(), img.cpu(), self.pattern, self.prior_size, 
                range(int(len(self.pattern)/2)), min(H, W))
            img = img.to(device)
            bb = bb.to(device)
            return img, bb[:,:5]

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
            if img_metas[i]['ori_filename'] in self.gt_bboxes_point.keys():
                # gt_points.append(self.gt_bboxes_point[img_metas[i]['ori_filename']])
                gt_points.append(self.refined_gt_bboxes_point[img_metas[i]['ori_filename']])
            else:
                if self._point_ == 'random':
                    random_points = random_point_in_quadrilateral(gt_bboxes[i], version=self.angle_version)
                elif self._point_ == 'center':
                    random_points = gt_bboxes[i][:,:2]
                gt_points.append(random_points)
                self.gt_bboxes_point[img_metas[i]['ori_filename']] = random_points
                self.refined_gt_bboxes_point[img_metas[i]['ori_filename']] = random_points
            img_list.append(img[i,:,:,:])
        return gt_points, img_list, img
               
    def forward_mil_head_burn_in_step1(self, num_img, synthetic_bboxes, pseudo_bboxes, pseudo_points, pseudo_labels, gt_bboxes, img_metas, 
                                       x_synthetic, x_ori):
        losses = {}
        syn_valid_inds = [i for i in range(num_img) if synthetic_bboxes[i].shape[0] != 0]
        if len(syn_valid_inds) != num_img:  
            return [torch.empty(0,5,device=pseudo_bboxes[0].device,dtype=pseudo_bboxes[0].dtype) for i in range(num_img)], \
                   [torch.empty(0,2,device=pseudo_bboxes[0].device,dtype=pseudo_bboxes[0].dtype) for i in range(num_img)], \
                   None
        synthetic_bboxes_training, synthetic_points_training, pseudo_bboxes_training, gt_bboxes_training, pseudo_labels_training, pseudo_points_training = [], [], [], [], [], []
        refined_pseudo_bboxes = []; refined_pseudo_points = []
        for i in range(num_img):
            synthetic_bboxes_training.append(synthetic_bboxes[i][:self.num_training_burninstep1,:].clone())
            synthetic_points_training.append(synthetic_bboxes_training[-1][:,:2].clone())
            pseudo_bboxes_training.append(pseudo_bboxes[i][:self.num_training_burninstep1,:].clone())
            gt_bboxes_training.append(gt_bboxes[i][:self.num_training_burninstep1,:].clone())
            pseudo_points_training.append(pseudo_points[i][:self.num_training_burninstep1,:].clone())
            pseudo_labels_training.append(pseudo_labels[i][:self.num_training_burninstep1].clone())
            refined_pseudo_bboxes.append(pseudo_bboxes[i].clone())
            refined_pseudo_points.append(pseudo_points[i].clone())
        coarse_bboxes_iou = rbbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                           torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
        losses['coarse_bboxes_iou'] = coarse_bboxes_iou
        for stage in range(self.num_stages):
            proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list = MIL_gen_proposals_from_cfg(pseudo_points_training, pseudo_bboxes_training, 
                                                                                                                             self.fine_proposal_cfg[stage], 
                                                                                                                             gt_bboxes_training, img_metas)
            syn_proposals_list, syn_proposals_valid_list, syn_proposals_reference_list, syn_proposals_real_list = MIL_gen_proposals_from_cfg(synthetic_points_training, synthetic_bboxes_training, 
                                                                                                                                             self.fine_proposal_cfg[stage],
                                                                                                                                             synthetic_bboxes_training, img_metas)
            neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_points_training, self.fine_proposal_cfg[stage],
                                                                        proposals_list, img_metas)
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
            refined_bboxes_iou = rbbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                                torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
            losses[f'stage{stage}_refine_bboxes_iou'] = refined_bboxes_iou
            mil_loss[f'stage{stage}_loss_mil_bbox'] = mil_loss[f'stage{stage}_loss_mil_bbox'] * self.alpha[0]
            mil_loss[f'stage{stage}_loss_mil_bags'] = mil_loss[f'stage{stage}_loss_mil_bags'] * self.alpha[1]
            losses.update(mil_loss)
        for i in range(num_img):
            refined_pseudo_bboxes[i][:self.num_training_burninstep1,:] = pseudo_bboxes_training[i]
            refined_pseudo_points[i][:self.num_training_burninstep1,:] = pseudo_bboxes_training[i][:,:2]
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
        coarse_bboxes_iou = rbbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                           torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
        losses['coarse_bboxes_iou'] = coarse_bboxes_iou
        for stage in range(self.num_stages):
            proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list = MIL_gen_proposals_from_cfg(pseudo_points_training, pseudo_bboxes_training, 
                                                                                                                             self.fine_proposal_cfg[stage], 
                                                                                                                             gt_bboxes_training, img_metas)
            neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_points_training, self.fine_proposal_cfg[stage],
                                                                        proposals_list, img_metas)
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
            refined_bboxes_iou = rbbox_overlaps(torch.cat(pseudo_bboxes_training, dim=0), 
                                                torch.cat(gt_bboxes_training, dim=0), mode='iou', is_aligned=True).mean()
            losses[f'stage{stage}_refine_bboxes_iou'] = refined_bboxes_iou
            mil_loss[f'stage{stage}_loss_mil_bbox'] = mil_loss[f'stage{stage}_loss_mil_bbox'] * self.alpha[0]
            mil_loss[f'stage{stage}_loss_mil_bags'] = mil_loss[f'stage{stage}_loss_mil_bags'] * self.alpha[1]
            losses.update(mil_loss)
        for i in range(num_img):
            refined_pseudo_bboxes[i][:self.num_training_burninstep2,:] = pseudo_bboxes_training[i]
            refined_pseudo_points[i][:self.num_training_burninstep2,:] = pseudo_bboxes_training[i][:,:2]
        return refined_pseudo_bboxes, refined_pseudo_points, losses