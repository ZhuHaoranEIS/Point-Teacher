from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 neck_agg,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet, self).__init__(backbone, neck, neck_agg, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        if neck_agg is not None:
            self.with_neck_agg = True
        else:
            self.with_neck_agg = False
