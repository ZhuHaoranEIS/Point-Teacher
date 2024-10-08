from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 neck_agg,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOS, self).__init__(backbone, neck, neck_agg, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        
    @property
    def with_neck_agg(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_agg') and self.neck_agg is not None
