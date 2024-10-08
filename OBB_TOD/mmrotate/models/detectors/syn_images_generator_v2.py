# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
import torchvision
from mmcv.ops import nms_rotated


import torch.nn.functional as F
import random
import math

from ...core.bbox.transforms import obb2poly, poly2obb, obb2poly_le90
import torchvision.transforms.functional as TF

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, distance2bbox, bbox2distance, 
                        build_assigner, build_sampler, bbox_overlaps)

from ...core.bbox.iou_calculators import rbbox_overlaps

EPS = 1e-2

def MIL_gen_proposals_from_cfg(pseudo_points, pseudo_boxes_obb, fine_proposal_cfg, gt_boxes_obb, img_meta):
    gen_model = fine_proposal_cfg['gen_mode']
    pseudo_boxes_hbb = [bbox_cxcywh_to_xyxy(pseudo_boxes_obb[i][:,:4]) for i in range(len(pseudo_boxes_obb))]
    if gen_model == 'refine':
        proposals_list, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes_hbb, fine_proposal_cfg, img_meta=img_meta)
    elif gen_model == 'define':
        proposals_list, proposals_valid_list = gen_proposals_from_cfg(pseudo_points, fine_proposal_cfg, img_meta)
    num_aug = int(proposals_list[0].shape[0] / pseudo_points[0].shape[0])
    proposals_reference_list, proposals_real_list = [], []
    for i in range(len(pseudo_boxes_obb)):
        proposals_reference_list.append(pseudo_boxes_obb[i].unsqueeze(1).repeat(1, num_aug, 1).reshape(-1, 5))
        proposals_real_list.append(gt_boxes_obb[i].unsqueeze(1).repeat(1, num_aug, 1).reshape(-1, 5))
        angle = pseudo_boxes_obb[i][:,-1].reshape(-1,1).unsqueeze(1).repeat(1, num_aug, 1).reshape(-1, 1)
        proposals_list[i] = torch.cat([bbox_xyxy_to_cxcywh(proposals_list[i]), angle], dim=1)
    return proposals_list, proposals_valid_list, proposals_reference_list, proposals_real_list

def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    if 'cut_mode' in proposal_cfg:
        cut_mode = proposal_cfg['cut_mode']
    else:
        cut_mode = 'symmetry'
    base_proposal_list = []
    proposals_valid_list = []
    for i in range(len(gt_points)):
        img_h, img_w, _ = img_meta[i]['img_shape']
        base = min(img_w, img_h) / 100
        base_proposals = []
        for scale in base_scales:
            scale = scale * base
            for ratio in base_ratios:
                base_proposals.append(gt_points[i].new_tensor([[scale * ratio, scale / ratio]]))

        base_proposals = torch.cat(base_proposals)
        base_proposals = base_proposals.repeat((len(gt_points[i]), 1))
        base_center = torch.repeat_interleave(gt_points[i], len(base_scales) * len(base_ratios), dim=0)

        if shake_ratio is not None:
            base_x_l = base_center[:, 0] - shake_ratio * base_proposals[:, 0]
            base_x_r = base_center[:, 0] + shake_ratio * base_proposals[:, 0]
            base_y_t = base_center[:, 1] - shake_ratio * base_proposals[:, 1]
            base_y_d = base_center[:, 1] + shake_ratio * base_proposals[:, 1]
            if cut_mode is not None:
                base_x_l = torch.clamp(base_x_l, 1, img_w - 1)
                base_x_r = torch.clamp(base_x_r, 1, img_w - 1)
                base_y_t = torch.clamp(base_y_t, 1, img_h - 1)
                base_y_d = torch.clamp(base_y_d, 1, img_h - 1)

            base_center_l = torch.stack([base_x_l, base_center[:, 1]], dim=1)
            base_center_r = torch.stack([base_x_r, base_center[:, 1]], dim=1)
            base_center_t = torch.stack([base_center[:, 0], base_y_t], dim=1)
            base_center_d = torch.stack([base_center[:, 0], base_y_d], dim=1)

            shake_mode = 0
            if shake_mode == 0:
                base_proposals = base_proposals.unsqueeze(1).repeat((1, 5, 1))
            elif shake_mode == 1:
                base_proposals_l = torch.stack([((base_center[:, 0] - base_x_l) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_r = torch.stack([((base_x_r - base_center[:, 0]) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_t = torch.stack([base_proposals[:, 0],
                                                ((base_center[:, 1] - base_y_t) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals_d = torch.stack([base_proposals[:, 0],
                                                ((base_y_d - base_center[:, 1]) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals = torch.stack(
                    [base_proposals, base_proposals_l, base_proposals_r, base_proposals_t, base_proposals_d], dim=1)

            base_center = torch.stack([base_center, base_center_l, base_center_r, base_center_t, base_center_d], dim=1)

        if cut_mode == 'symmetry':
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * base_center[..., 0])
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * (img_w - base_center[..., 0]))
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * base_center[..., 1])
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * (img_h - base_center[..., 1]))

        base_proposals = torch.cat([base_center, base_proposals], dim=-1)
        base_proposals = base_proposals.reshape(-1, 4)
        base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
        proposals_valid = base_proposals.new_full(
            (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        if cut_mode == 'clamp':
            base_proposals[..., 0:4:2] = torch.clamp(base_proposals[..., 0:4:2], 0, img_w)
            base_proposals[..., 1:4:2] = torch.clamp(base_proposals[..., 1:4:2], 0, img_h)
            proposals_valid_list.append(proposals_valid)
        if cut_mode == 'symmetry':
            proposals_valid_list.append(proposals_valid)
        elif cut_mode == 'ignore':
            img_xyxy = base_proposals.new_tensor([0, 0, img_w, img_h])
            iof_in_img = bbox_overlaps(base_proposals, img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
        elif cut_mode is None:
            proposals_valid_list.append(proposals_valid)
        base_proposal_list.append(base_proposals)

    return base_proposal_list, proposals_valid_list


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        # x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        # y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        # x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        # y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        x1 = torch.rand(num_neg_gen) * w * 0.8
        y1 = torch.rand(num_neg_gen) * h * 0.8
        x2 = x1 + torch.rand(num_neg_gen) * 200
        y2 = y1 + torch.rand(num_neg_gen) * 200
        theta = torch.rand(num_neg_gen) * torch.pi - torch.pi / 2

        neg_bboxes = torch.stack([x1, y1, x2, y2, theta], dim=1).to(gt_points[0].device)
        gt_point = gt_points[i]
        gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
        iou = rbbox_overlaps(neg_bboxes, pos_box)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    return neg_proposal_list, neg_weight_list


def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    min_scale = fine_proposal_cfg['min_scale']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    base_ratios = fine_proposal_cfg['base_ratios']
    shake_ratio = fine_proposal_cfg['shake_ratio']
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
                    base_boxes_[:, 2] = base_boxes_[:, 2].clamp(min_scale, 1000)
                    base_boxes_[:, 3] = base_boxes_[:, 3].clamp(min_scale, 1000)
                    base_boxes_[:, 2] *= ratio_w
                    base_boxes_[:, 3] *= ratio_h
                    # print(base_boxes_)
                    base_boxes_ = bbox_cxcywh_to_xyxy(base_boxes_)
                    pps.append(base_boxes_.unsqueeze(1))
            pps_old = torch.cat(pps, dim=1)
            if shake_ratio is not None:
                pps_new = []
                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
                for ratio in shake_ratio:
                    pps = bbox_xyxy_to_cxcywh(pps_old)
                    pps_center = pps[:, :, :2]
                    pps_wh = pps[:, :, 2:4]
                    pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                    pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                    pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
                    pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
                    pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                    pps = torch.cat([pps_center, pps_wh], dim=-1)
                    pps = pps.reshape(pps.shape[0], -1, 4)
                    pps = bbox_cxcywh_to_xyxy(pps)
                    pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            h, w, _ = img_meta[i]['img_shape']
            if cut_mode is 'clamp':
                pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
                pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
                proposals_valid_list.append(pps_new.new_full(
                    (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            else:
                img_xyxy = pps_new.new_tensor([0, 0, w, h])
                iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
                proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)

            proposal_list.append(pps_new.reshape(-1, 4))

    return proposal_list, proposals_valid_list

def strong_augmentation(img, gt_points, gt_labels, pseudo_points, pseudo_labels, pseudo_bboxes, angle_version):
    B, C, H, W = img.shape
    img_aug_list = []
    aug_gt_points_list = []
    aug_gt_labels_list = []
    aug_pseudo_bboxes_list = []
    aug_pseudo_labels_list = []
    aug_pseudo_points_list = []
    for i in range(B):
        img_aug = img[i,:,:,:].clone()
        aug_gt_labels = gt_labels[i].clone()
        aug_gt_points = gt_points[i].clone()
        aug_pseudo_bboxes = obb2poly(pseudo_bboxes[i].clone(), version=angle_version)
        aug_pseudo_labels = pseudo_labels[i].clone()
        aug_pseudo_points = pseudo_points[i].clone()
        ### random flip
        valid_directions = ['horizontal', 'vertical', 'diagonal', 'None']
        flip_chosen = random.choice(valid_directions)
        if flip_chosen == 'horizontal':
            img_aug = torch.flip(img_aug, dims=[2])
            aug_pseudo_bboxes[:,0::2] = W - aug_pseudo_bboxes[:,0::2]
            aug_pseudo_points[:,0] = W - aug_pseudo_points[:,0]
            aug_gt_points[:,0] = W - aug_gt_points[:,0]
        elif flip_chosen == 'vertical':
            img_aug = torch.flip(img_aug, dims=[1])
            aug_pseudo_bboxes[:,1::2] = H - aug_pseudo_bboxes[:,1::2]
            aug_pseudo_points[:,1] = H - aug_pseudo_points[:,1]
            aug_gt_points[:,1] = H - aug_gt_points[:,1]
        elif flip_chosen == 'diagonal':
            img_aug = torch.flip(img_aug, dims=[1,2])
            aug_pseudo_bboxes[:,0::2] = W - aug_pseudo_bboxes[:,0::2]
            aug_pseudo_bboxes[:,1::2] = H - aug_pseudo_bboxes[:,1::2]
            aug_pseudo_points[:,0] = W - aug_pseudo_points[:,0]
            aug_pseudo_points[:,1] = H - aug_pseudo_points[:,1]
            aug_gt_points[:,0] = W - aug_gt_points[:,0]
            aug_gt_points[:,1] = H - aug_gt_points[:,1]
        else:
            img_aug = img_aug
            
        ### random rotate
        angle = np.random.randint(1, 20)
        cx, cy = W / 2, H / 2 
        img_aug = TF.rotate(img_aug, angle, fill=0)
        
        radians = np.deg2rad(-angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        temp_aug_pseudo_bboxes = aug_pseudo_bboxes.clone()
        temp_aug_pseudo_points = aug_pseudo_points.clone()
        temp_aug_gt_points = aug_gt_points.clone()
        aug_pseudo_bboxes[:,0::2] = cos_a * (temp_aug_pseudo_bboxes[:,0::2] - cx) - sin_a * (temp_aug_pseudo_bboxes[:,1::2] - cy) + cx
        aug_pseudo_bboxes[:,1::2] = sin_a * (temp_aug_pseudo_bboxes[:,0::2] - cx) + cos_a * (temp_aug_pseudo_bboxes[:,1::2] - cy) + cy
        
        aug_pseudo_points[:,0] = cos_a * (temp_aug_pseudo_points[:,0] - cx) - sin_a * (temp_aug_pseudo_points[:,1] - cy) + cx
        aug_pseudo_points[:,1] = sin_a * (temp_aug_pseudo_points[:,0] - cx) + cos_a * (temp_aug_pseudo_points[:,1] - cy) + cy
        
        aug_gt_points[:,0] = cos_a * (temp_aug_gt_points[:,0] - cx) - sin_a * (temp_aug_gt_points[:,1] - cy) + cx
        aug_gt_points[:,1] = sin_a * (temp_aug_gt_points[:,0] - cx) + cos_a * (temp_aug_gt_points[:,1] - cy) + cy
        del temp_aug_pseudo_bboxes; del temp_aug_pseudo_points; del temp_aug_gt_points
        
        insider_gt = (((0 <= aug_gt_points[:,0]) & (aug_gt_points[:,0] < W)) & ((0 <= aug_gt_points[:,1]) & (aug_gt_points[:,1] < H))).nonzero().reshape(-1)
        aug_gt_points = aug_gt_points[insider_gt,:]
        aug_gt_labels = aug_gt_labels[insider_gt]
        
        insider_pseudo = (((0 <= aug_pseudo_points[:,0]) & (aug_pseudo_points[:,0] < W)) & ((0 <= aug_pseudo_points[:,1]) & (aug_pseudo_points[:,1] < H))).nonzero().reshape(-1)
        aug_pseudo_points = aug_pseudo_points[insider_pseudo,:]
        aug_pseudo_labels = aug_pseudo_labels[insider_pseudo]
        aug_pseudo_bboxes = aug_pseudo_bboxes[insider_pseudo,:]
        
        ### random rescale
        scale_factor = np.around(np.random.uniform(0.8, 1.2), 1)
        scale_H, scale_W = int(H*scale_factor), int(W*scale_factor)
        if scale_factor < 1.0:
            blank_h = int((H - scale_H) / 2)
            blank_w = int((W - scale_W) / 2)
        else:
            blank_h = int((scale_H - H) / 2)
            blank_w = int((scale_W - W) / 2)
        
        aug_pseudo_bboxes *= scale_factor
        aug_pseudo_points *= scale_factor
        aug_gt_points *= scale_factor
        
        if scale_factor >= 1.0:
            insider_gt = ((aug_gt_points[:, 0] >= (blank_w)) & (aug_gt_points[:, 0] < (W+blank_w)) &
                            (aug_gt_points[:, 1] >= (blank_h)) & (aug_gt_points[:, 1] < (H+blank_h))).nonzero().reshape(-1)
            aug_gt_points = aug_gt_points[insider_gt,:]
            aug_gt_labels = aug_gt_labels[insider_gt]
            aug_gt_points[:,0] -= blank_w; aug_gt_points[:,1] -= blank_h
            
            insider_pseudo = ((aug_pseudo_points[:, 0] >= (blank_w)) & (aug_pseudo_points[:, 0] < (W+blank_w)) &
                            (aug_pseudo_points[:, 1] >= (blank_h)) & (aug_pseudo_points[:, 1] < (H+blank_h))).nonzero().reshape(-1)
            aug_pseudo_bboxes = aug_pseudo_bboxes[insider_pseudo,:]
            aug_pseudo_points = aug_pseudo_points[insider_pseudo,:]
            aug_pseudo_labels = aug_pseudo_labels[insider_pseudo]
            aug_pseudo_points[:,0] -= blank_w; aug_pseudo_points[:,1] -= blank_h
            aug_pseudo_bboxes[:,0::2] -= blank_w; aug_pseudo_bboxes[:,1::2] -= blank_h
        else:
            aug_gt_points[:,0] += blank_w; aug_gt_points[:,1] += blank_h
            aug_pseudo_points[:,0] += blank_w; aug_pseudo_points[:,1] += blank_h
            aug_pseudo_bboxes[:,0::2] += blank_w; aug_pseudo_bboxes[:,1::2] += blank_h
        
        rescaled_image = F.interpolate(img_aug.unsqueeze(0), size=(scale_H, scale_W), mode='bilinear', align_corners=False).squeeze(0)
        img_aug = torch.zeros_like(img_aug)
        if scale_factor < 1.0:
            start_y = (H - scale_H) // 2
            end_y = start_y + scale_H
            start_x = (W - scale_W) // 2
            end_x = start_x + scale_W
            img_aug[:, start_y:end_y, start_x:end_x] = rescaled_image
        else:
            start_y = (scale_H - H) // 2
            end_y = start_y + H
            start_x = (scale_W - W) // 2
            end_x = start_x + W
            img_aug = rescaled_image[:, start_y:end_y, start_x:end_x]
        
        img_aug = torch.round(img_aug)
        
        ### bboxes refine
        if len(aug_pseudo_bboxes) != 0:
            aug_pseudo_bboxes = poly2obb(aug_pseudo_bboxes, version=angle_version)
        else:
            aug_pseudo_bboxes = torch.empty(0, 5).to(gt_points[0].device).to(gt_points[0].dtype)
        
        img_aug_list.append(img_aug)
        aug_pseudo_bboxes_list.append(aug_pseudo_bboxes)
        aug_pseudo_labels_list.append(aug_pseudo_labels)
        aug_pseudo_points_list.append(aug_pseudo_points)
        
        aug_gt_points_list.append(aug_gt_points)
        aug_gt_labels_list.append(aug_gt_labels)
    aug_images = torch.stack(img_aug_list, dim=0)
        
    return aug_images, img_aug_list, aug_gt_points_list, aug_gt_labels_list, \
            aug_pseudo_points_list, aug_pseudo_labels_list, aug_pseudo_bboxes_list


def get_pattern_fill(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    cv2.line(p, (0, 0), (0, h - 1), 0.01, 1)
    cv2.line(p, (0, 0), (w - 1, 0), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (0, h - 1), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (w - 1, 0), 0.01, 1)
    return p


def get_pattern_line(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    interval_range = [3, 6]
    xn = np.random.randint(*interval_range)
    yn = np.random.randint(*interval_range)
    for i in range(xn):
        x = np.intp(np.round((w - 1) * i / (xn - 1)))
        cv2.line(p, (x, 0), (x, h), 0.5, 1)
    for i in range(yn):
        y = np.intp(np.round((h - 1) * i / (yn - 1)))
        cv2.line(p, (0, y), (w, y), 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_rose(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    omega_range = [2, 4]
    xn = np.random.randint(*omega_range)
    x = np.sin(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * xn) * w / 2 + w / 2
    y = np.cos(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * 2) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_li(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    s = np.random.rand() * 8
    s2 = np.random.rand() * 0.5 + 0.1
    r = (np.abs(np.cos(t / 99 * 4 * np.pi))**s) * (1 - s2) + s2
    x = r * np.sin(t / 99 * 2 * np.pi) * w / 2 + w / 2
    y = r * np.cos(t / 99 * 2 * np.pi) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.fillPoly(p, np.int_([xy]), 1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def obb2xyxy(obb):
    w = obb[:, 2]
    h = obb[:, 3]
    a = obb[:, 4]
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    dw = cosa * w + sina * h
    dh = sina * w + cosa * h
    dx = obb[..., 0]
    dy = obb[..., 1]
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)


def get_pattern_gaussian(w, h, device):
    w, h = int(w), int(h)
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device))
    y = (y - h / 2) / (h / 2)
    x = (x - w / 2) / (w / 2)
    ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
    sx, sy = torch.rand(2, device=device) + 0.3
    z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.9 + 0.1
    return z

def generate_sythesis(img, bb_occupied, img_syn, sca_fact, pattern, prior_size,
                      dense_cls, imgsize, scale_ratio=0.5):  
    torch.pi = math.pi  
    device = img.device
    cen_range = [50, imgsize - 50]

    scale_vary = torch.rand(bb_occupied.shape[0]).to(device) * 1.8 + 0.2 # 0.2 -- 2.0

    bb_occupied = bb_occupied.clone()
    bb_occupied[:, 2] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 3] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 4] = 0

    bb = []
    palette = [
        torch.zeros(0, 6, device=device) for _ in range(len(prior_size))
    ]
    adjboost = 2
    count_all = 0
    for b in bb_occupied:
        base_scale = scale_vary[count_all]
        count_all += 1

        x, y = torch.rand(
            2, device=device) * (cen_range[1] - cen_range[0]) + cen_range[0]
        
        dw = prior_size[b[6].long(), 2]
        w = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dw
        w = base_scale * torch.exp(w)
        dr = prior_size[b[6].long(), 3]
        r = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dr

        h = w * torch.exp(r)
        w *= prior_size[b[6].long(), 0]
        h *= prior_size[b[6].long(), 1]
        
        a = torch.rand(1, device=device) * torch.pi - torch.pi / 2

        x = x.clip(0.71 * w, imgsize - 1 - 0.71 * w)
        y = y.clip(0.71 * h, imgsize - 1 - 0.71 * h)
        bb.append([x, y, w, h, a, (w * h) / imgsize / imgsize + 0.1, b[6]])

        bx = torch.clip(b[0:2], 16, imgsize - 1 - 16).long()
        nbr0 = img[:, bx[1] - 2:bx[1] + 3, bx[0] - 2:bx[0] + 3].reshape(3, -1)
        nbr1 = img[:, bx[1] - 16:bx[1] + 17,
                   bx[0] - 16:bx[0] + 17].reshape(3, -1)
        c0 = nbr0.mean(1)
        c1 = nbr1[:, (nbr1.mean(0) - c0.mean()).abs().max(0)[1]]
        c = torch.cat((c0, c1), 0)
        palette[b[6].long()] = torch.cat((palette[b[6].long()], c[None]), 0)

        if np.random.random() < 0.2 and adjboost > 0:
            adjboost -= 1
            if b[6].long() in dense_cls:
                itv, dev = torch.rand(
                    1, device=device) * 4 + 2, torch.rand(
                        1, device=device) * 8 - 4
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 6):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])
            else:
                itv, dev = torch.rand(
                    1, device=device) * 40 + 10, torch.rand(
                        1, device=device) * 0
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 4):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])

    bb = torch.tensor(bb)
    bb = torch.cat((bb_occupied, bb), 0)
    _, keep = nms_rotated(bb[:, 0:5], bb[:, 5], 0.05)
    bb = bb[keep]
    bb = bb[bb[:, 5] < 1]

    xyxy = obb2xyxy(bb)
    mask = torch.logical_and(
        xyxy.min(-1)[0] >= 0,
        xyxy.max(-1)[0] <= imgsize - 1)
    bb, xyxy = bb[mask], xyxy[mask]

    for i in range(len(bb)):
        cx, cy, w, h, t, s, c = bb[i, :7]
        ox, oy = torch.floor(xyxy[i, 0:2]).long()
        ex, ey = torch.ceil(xyxy[i, 2:4]).long()
        sx, sy = ex - ox, ey - oy
        theta = torch.tensor(
            [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
             [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]],
            dtype=torch.float,
            device=device)
        theta[:, :2] @= torch.tensor([[sx, 0], [0, sy]],
                                     dtype=torch.float,
                                     device=device)
        grid = torch.nn.functional.affine_grid(
            theta[None], (1, 1, sy, sx), align_corners=False)
        p = pattern[c.long()]
        p = p[np.random.randint(0, len(p))][None].clone()
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])
        if np.random.random() < 0.2:
            p *= get_pattern_line(p.shape[2], p.shape[1])
        if np.random.random() < 0.2:
            p *= get_pattern_rose(p.shape[2], p.shape[1])
        if np.random.random() < 0.2:
            p *= get_pattern_li(p.shape[2], p.shape[1])
        p = trans(p)
        p = torch.nn.functional.grid_sample(
            p[None], grid, align_corners=False, mode='nearest')[0]
        if np.random.random() < 0.9:
            a = get_pattern_gaussian(sx, sy, device) * (p != 0)
        else:
            a = (p != 0).float()
        pal = palette[c.long()]
        color = pal[torch.randint(0, len(pal), (1, ))][0]
        p = p * color[:3, None, None] + (1 - p) * color[3:, None, None]
        # img_syn[:, oy:oy + sy,
        #     ox:ox + sx] = (1 - a) * img_syn[:, oy:oy + sy, ox:ox + sx] + a * p

        p_randn = torch.randint_like(p, low=0, high=255) * 0
        img_syn[:, oy:oy + sy, ox:ox + sx] = p_randn

    return img_syn, bb


def load_basic_pattern(path, set_rc=True, set_sk=True):
    with open(os.path.join(path, 'properties.txt')) as f:
        prior = eval(f.read())
    prior_size = torch.tensor(list(prior.values()))
    pattern = []
    for i in range(len(prior_size)):
        p = []
        if set_rc:
            img = get_pattern_fill(*prior_size[i, (0, 1)])
            p.append(torch.from_numpy(img).float())
        if set_sk:
            if os.path.exists(os.path.join(path, f'{i}.png')):
                img = cv2.imread(os.path.join(path, f'{i}.png'), 0)
                img = img / 255
            else:
                img = get_pattern_fill(*prior_size[i, (0, 1)])
            p.append(torch.from_numpy(img).float())
        pattern.append(p)
    return pattern, prior_size


def load_basic_shape(shape_list):
    torch.pi = math.pi 
    # shape_list = [[20, 20, 0.5, 0.5], [30, 120, 0.5, 0.5], [10, 20, 0.5, 0.5],
    #               [20, 50, 0.5, 0.5], [30, 20, 0.5, 0.5]]
    prior_size = torch.Tensor(shape_list).float()
    pattern = []
    for i in range(len(shape_list)):
        pattern.append([torch.zeros(shape_list[i][:2]).float()])
    return pattern, prior_size

def generate_black_paper(img, bb_occupied, img_syn, pattern, prior_size,
                        dense_cls, imgsize):  
    device = img.device
    cen_range = [50, imgsize - 50]
    C, H, W = img.shape

    scale_vary = torch.rand(bb_occupied.shape[0]).to(device) * 2.0 + 0.5 # 0.2 -- 2.0

    bb_occupied = bb_occupied.clone()
    bb_occupied[:, 2] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 3] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 4] = 0

    bb = []
    palette = [
        torch.zeros(0, 6, device=device) for _ in range(len(prior_size))
    ]
    adjboost = 2
    count_all = 0
    for b in bb_occupied:
        base_scale = scale_vary[count_all]
        count_all += 1
        x, y = torch.rand(
            2, device=device) * (cen_range[1] - cen_range[0]) + cen_range[0]
        dw = prior_size[b[6].long(), 2]
        w = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dw
        w = base_scale * torch.exp(w)
        dr = prior_size[b[6].long(), 3]
        r = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dr

        h = w * torch.exp(r)
        w *= prior_size[b[6].long(), 0]
        h *= prior_size[b[6].long(), 1]
        
        a = torch.rand(1, device=device) * torch.pi - torch.pi / 2

        x = x.clip(0.71 * w, imgsize - 1 - 0.71 * w)
        y = y.clip(0.71 * h, imgsize - 1 - 0.71 * h)
        bb.append([x, y, w, h, a, (w * h) / imgsize / imgsize + 0.1, b[6]])

        bx = torch.clip(b[0:2], 16, imgsize - 1 - 16).long()
        nbr0 = img[:, bx[1] - 2:bx[1] + 3, bx[0] - 2:bx[0] + 3].reshape(3, -1)
        nbr1 = img[:, bx[1] - 16:bx[1] + 17,
                   bx[0] - 16:bx[0] + 17].reshape(3, -1)
        c0 = nbr0.mean(1)
        c1 = nbr1[:, (nbr1.mean(0) - c0.mean()).abs().max(0)[1]]
        c = torch.cat((c0, c1), 0)
        palette[b[6].long()] = torch.cat((palette[b[6].long()], c[None]), 0)

        if np.random.random() < 0.2 and adjboost > 0:
            adjboost -= 1
            if b[6].long() in dense_cls:
                itv, dev = torch.rand(
                    1, device=device) * 4 + 2, torch.rand(
                        1, device=device) * 8 - 4
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 6):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])
            else:
                itv, dev = torch.rand(
                    1, device=device) * 40 + 10, torch.rand(
                        1, device=device) * 0
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 4):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])

    bb = torch.tensor(bb)
    bb = torch.cat((bb_occupied, bb), 0)
    _, keep = nms_rotated(bb[:, 0:5], bb[:, 5], 0.05)
    bb = bb[keep]
    bb = bb[bb[:, 5] < 1]

    xyxy = obb2xyxy(bb)
    mask = torch.logical_and(
        xyxy.min(-1)[0] >= 0,
        xyxy.max(-1)[0] <= imgsize - 1)
    bb, xyxy = bb[mask], xyxy[mask]

    
    polygons = obb2poly_le90(bb[:,:5])
    mask = np.zeros((H, W), dtype=np.uint8)
    polygons = polygons.view(-1, 4, 2).numpy()
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    mask = torch.tensor(mask).to(img.dtype).to(img.device)
    mask = mask.unsqueeze(0).expand_as(img)
    # img_syn = img * (1 - mask)
    poly_area_inds = (mask == 1).nonzero()
    # import pdb; pdb.set_trace()
    img_syn[poly_area_inds[:,0],poly_area_inds[:,1], poly_area_inds[:,2]] = img_syn.max()

    return img_syn, bb



if __name__ == '__main__':
    pattern, prior_size = load_basic_pattern(
        '/home/zhuhaoran/point2rbox/data/basic_patterns/dota')

    img = cv2.imread('/data/zhr/SODA/SODA-A/divData/train/Images_filter/00001__800__3900___0.jpg')
    img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous()
    # import pdb; pdb.set_trace()
    bb = torch.tensor(((318, 84, 0, 0, 0.5, 17, 19), ) * 10)
    bb[:,-1] = torch.tensor(range(0,bb.shape[0]))
    import time
    c = time.time()
    # img, bb1 = generate_sythesis(img, bb, 1)
    # print(time.time() - c)
    # bb1[:, 5] = 1
    c = time.time()
    img, bb2 = generate_sythesis(img, bb, img, 1.0, pattern, prior_size, range(len(pattern)), 800, scale_ratio=0.5)
    # import pdb; pdb.set_trace()
    print(time.time() - c)
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()

    cv2.imwrite('/home/zhuhaoran/SODA/mmrotate/models/detectors/vis_points/P2RB_sys.png', img)

    coordinates_flip = bb2[:,:5]

    # print("Original Image Shape:", image.shape)
    # print("Scaled Image Shape:", flip_image.shape)
    ### 地面
    base_path = '/data/zhr/SODA/SODA-A/divData/train/Images_filter/'
    '00299__800__650___1950.jpg'
    '00302__800__1300___1300.jpg'
    '01423__800__1300___650.jpg'
    '01752__800__3900___650.jpg'
    '00520__800__3250___650.jpg'
    '00108__800__0___650.jpg'
    ### 海面
    base_path = '/data/xuchang/AI-TOD/train/'
    '7a03dc1c7.png'
    '7a9708edd.png'
    '7ac3a346a.png'
    '7ba49f141.png'
    '7beed9136.png'
    '7c4b06990.png'
    


