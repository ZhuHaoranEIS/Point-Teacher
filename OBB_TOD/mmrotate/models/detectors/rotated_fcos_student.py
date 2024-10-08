# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv import ConfigDict
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck, build_detector

# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector

from mmrotate.core import rbbox2result, build_bbox_coder
from ..builder import build_backbone, build_head, build_neck
from ...core.bbox.transforms import obb2poly
import torch
import numpy as np
import cv2

from ..builder import build_loss

from mmdet.core import multi_apply, reduce_mean

from .data_augument_bank import random_point_in_quadrilateral, random_rescale_image, random_rotate_image, \
                                random_translate_image, random_flip_image, save_image, load_image, imshow_det_rbboxes, \
                                fix_rescale_rbbox, fix_rotate_rbbox, fix_translate_rbbox, fix_flip_rbbox

from .z_transformer import distribution_ssl_orientation_loss

from .point2rbox_generator import load_basic_pattern, generate_sythesis

from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, distance2bbox, bbox2distance, 
                        build_assigner, build_sampler, bbox_overlaps)

import torch.nn.functional as F
import math
import time

import random


@ROTATED_DETECTORS.register_module()
class RotatedFCOS_Student(RotatedSingleStageDetector):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def __init__(self,
                 backbone,
                 neck,
                 neck_agg,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedFCOS_Student, self).__init__(backbone, neck, neck_agg, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)

    @property
    def with_neck_agg(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_agg') and self.neck_agg is not None
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if self.with_neck_agg:
            x = self.neck_agg(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img, self.backbone, self.neck, self.neck_agg)
        outs = self.bbox_head(x)
        return outs
    
