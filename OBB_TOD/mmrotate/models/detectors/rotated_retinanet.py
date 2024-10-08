# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedRetinaNet(RotatedSingleStageDetector):
    """Implementation of Rotated `RetinaNet.`__

    __ https://arxiv.org/abs/1708.02002
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
        super(RotatedRetinaNet,
              self).__init__(backbone, neck, neck_agg, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)
        # if neck_agg == None:
        #     self.with_neck_agg = False
        # else:
        #     self.with_neck_agg = True
