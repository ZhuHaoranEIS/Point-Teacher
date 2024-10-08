from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector

import torch

from mmdet.core import bbox2result


@DETECTORS.register_module()
class Student_FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 neck_agg,
                 bbox_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Student_FCOS, self).__init__(backbone, neck, neck_agg, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        if roi_head is not None:
            self.roi_head = build_head(roi_head)
        else:
            self.roi_head = None

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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
