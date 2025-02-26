import torch
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .builder import MATCH_COST


@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

@MATCH_COST.register_module()
class PointCost:
    
    def __init__(self, mode='L1', weight=1.):
        assert mode in ['L1', 'L2']
        self.weight = weight
        self.mode = mode

    def __call__(self, bboxes, gt_bboxes):
        '''
        bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
        bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, shape (m, 6) in
                <cx, cy, w, h, a, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
        '''
        # overlaps: [num_bboxes, num_gt]
        points1 = bboxes[:,:2]
        points2 = gt_bboxes[:,:2]
        points1 = points1.unsqueeze(1)  # Shape: (m, 1, 2)
        points2 = points2.unsqueeze(0)  # Shape: (1, n, 2)
        if self.mode == 'L1':
            distances = torch.sum(torch.abs(points1 - points2), dim=2)  # L1 distance
        elif self.mode == 'L2':
            distances = torch.sqrt(torch.sum((points1 - points2) ** 2, dim=2))  # L2 distance
        return distances * self.weight
    
@MATCH_COST.register_module()
class InsiderCost:
    
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes):
        '''
        bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
        bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, shape (m, 6) in
                <cx, cy, w, h, a, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
        '''
        # overlaps: [num_bboxes, num_gt]
        M = bboxes.shape[0]
        N = gt_bboxes.shape[0]
        boxes = bboxes[:,:4]   # M 4
        gt_points = gt_bboxes[:,:2]# N 2
        
        boxes_x1 = boxes[:, 0] - boxes[:, 2] / 2
        boxes_y1 = boxes[:, 1] - boxes[:, 3] / 2
        boxes_x2 = boxes[:, 0] + boxes[:, 2] / 2
        boxes_y2 = boxes[:, 1] + boxes[:, 3] / 2

        gt_points_x = gt_points[:, 0].unsqueeze(1).expand(N, M)
        gt_points_y = gt_points[:, 1].unsqueeze(1).expand(N, M)

        in_box_x = (gt_points_x >= boxes_x1.unsqueeze(0)) & (gt_points_x <= boxes_x2.unsqueeze(0))
        in_box_y = (gt_points_y >= boxes_y1.unsqueeze(0)) & (gt_points_y <= boxes_y2.unsqueeze(0))
        in_box = in_box_x & in_box_y

        cost_matrix = torch.where(in_box, torch.zeros_like(in_box), torch.ones_like(in_box)).permute(1,0)
        
        return cost_matrix * self.weight
    
@MATCH_COST.register_module()
class CenternessCost:
    
    def __init__(self, mode='L1', weight=1.):
        assert mode in ['L1', 'L2']
        self.weight = weight
        self.mode = mode

    def __call__(self, centerness, gt_centerness):
        '''
        bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
        bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, shape (m, 6) in
                <cx, cy, w, h, a, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
        '''
        # overlaps: [num_bboxes, num_gt]
        points1 = centerness.reshape(-1,1)
        points2 = gt_centerness.reshape(-1,1)
        points1 = points1.unsqueeze(1)  # Shape: (m, 1, 2)
        points2 = points2.unsqueeze(0)  # Shape: (1, n, 2)
        if self.mode == 'L1':
            distances = torch.sum(torch.abs(points1 - points2), dim=2)  # L1 distance
        elif self.mode == 'L2':
            distances = torch.sqrt(torch.sum((points1 - points2) ** 2, dim=2))  # L2 distance
        return distances * self.weight
    
@MATCH_COST.register_module()
class SAMPointCost:
    
    def __init__(self, mode='L1', weight=1.):
        assert mode in ['L1', 'L2']
        self.weight = weight
        self.mode = mode

    def __call__(self, bboxes, gt_bboxes):
        '''
        bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
        bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, shape (m, 6) in
                <cx, cy, w, h, a, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
        '''
        # overlaps: [num_bboxes, num_gt]
        points1 = bboxes
        points2 = gt_bboxes
        points1 = points1.unsqueeze(1)  # Shape: (m, 1, k)
        points2 = points2.unsqueeze(0)  # Shape: (1, n, k)
        if self.mode == 'L1':
            distances = torch.sum(torch.abs(points1 - points2), dim=2)  # L1 distance
        elif self.mode == 'L2':
            distances = torch.sqrt(torch.sum((points1 - points2) ** 2, dim=2))  # L2 distance
        return distances * self.weight
    
@MATCH_COST.register_module()
class HPointCost:
    
    def __init__(self, mode='L1', weight=1.):
        assert mode in ['L1', 'L2']
        self.weight = weight
        self.mode = mode

    def xyxy2cxcywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
        return bboxes[:,:2]

    def __call__(self, bboxes, gt_bboxes):
        # overlaps: [num_bboxes, num_gt]
        points1 = self.xyxy2cxcywh(bboxes)
        points2 = self.xyxy2cxcywh(gt_bboxes)
        
        points1 = points1.unsqueeze(1)  # Shape: (m, 1, 2)
        points2 = points2.unsqueeze(0)  # Shape: (1, n, 2)
        if self.mode == 'L1':
            distances = torch.sum(torch.abs(points1 - points2), dim=2)  # L1 distance
        elif self.mode == 'L2':
            distances = torch.sqrt(torch.sum((points1 - points2) ** 2, dim=2))  # L2 distance
        return distances * self.weight
    
@MATCH_COST.register_module()
class CrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    Examples:
         >>> from mmdet.core.bbox.match_costs import CrossEntropyLossCost
         >>> import torch
         >>> bce = CrossEntropyLossCost(use_sigmoid=True)
         >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
         >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
         >>> print(bce(cls_pred, gt_labels))
    """

    def __init__(self, weight=1., use_sigmoid=True):
        assert use_sigmoid, 'use_sigmoid = False is not supported yet.'
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.

        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight

