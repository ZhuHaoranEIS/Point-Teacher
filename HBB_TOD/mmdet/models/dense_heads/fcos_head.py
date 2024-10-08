import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


INF = 1e8


@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

        # visual 
        self.feats_all = {}
        self.num_all = {}
        for i in range(self.num_classes):
            self.feats_all[i] = np.ones((5000, self.feat_channels))
            self.num_all[i] = 0


    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        '''
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            #stacked_convs=4, 即接4个3*3的卷积, channel为256
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        '''
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        # 传进来的feats就是通过backbone和FPN提取多尺度的特征(一般是一个长度为５的tuple)
        # 注意这边用map(即multi_apply)把FPN的多个level给拆解了，可以简单理解为并行计算了
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        # 所以x其实是FPN的一个Level的特征
        cls_feat = x
        reg_feat = x
        # 把输入特征分别前向过4个卷积(上一分岔路)
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        #过分类对应的卷积层,输出是H*W*(numcls-1)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat
        """
        #过centerness对应的卷积层,输出是H*W*1
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        # 过回归任务对应的卷积层,输出是H*W*４，注意最后跟了exp函数映射了值
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        # 返回值就是算出的这个三个部分分类回归和centerness
        # 需要注意的是，之后multi_apply后返回的是和FPN尺度对应的多尺度的这些结果
        return cls_score, bbox_pred, centerness, cls_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'cls_feats'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             cls_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # 每一层的feature大小被得到，[-2：-1]为H*W

        # This is a very essential problem, but why and what is this?
        # the dtype and device are used to avoid the same operation from different CUDA 
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        '''
        >>> points (if h, w equals 5, 10; then all_level_points equals follow)
        tensor([[0., 0.],
        [1., 0.],
        [2., 0.],
        [3., 0.],
        [4., 0.],
        [0., 1.],
        [1., 1.],
        [2., 1.],
        [3., 1.],
        [4., 1.],
        [0., 2.],
        [1., 2.],
        [2., 2.],
        [3., 2.],
        [4., 2.],
        [0., 3.],
        [1., 3.],
        [2., 3.],
        [3., 3.],
        [4., 3.],
        [0., 4.],
        [1., 4.],
        [2., 4.],
        [3., 4.],
        [4., 4.],
        [0., 5.],
        [1., 5.],
        [2., 5.],
        [3., 5.],
        [4., 5.],
        [0., 6.],
        [1., 6.],
        [2., 6.],
        [3., 6.],
        [4., 6.],
        [0., 7.],
        [1., 7.],
        [2., 7.],
        [3., 7.],
        [4., 7.],
        [0., 8.],
        [1., 8.],
        [2., 8.],
        [3., 8.],
        [4., 8.],
        [0., 9.],
        [1., 9.],
        [2., 9.],
        [3., 9.],
        [4., 9.]])
        '''

        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # cls_scores的每个元素是FPN的每个level的分类预测结果
        # 如Size为([4, 80, 96, 168])代表NCHW,C为类别数目－１
        # permute后变为([4, 96, 168, 80]),reshape后变为(64512, 80)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_feats = [
            cls_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)
            for cls_feat in cls_feats
        ]
        flatten_cls_feats = torch.cat(flatten_cls_feats)

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        

        ### visual ###
        '''
        for i in range(self.num_classes):
            k = (flatten_labels == i).nonzero().reshape(-1)
            if k.numel() == 0:
                continue
            for j in range(k.numel()):
                self.feats_all[i][self.num_all[i]%5000,:] = flatten_cls_feats[k[j],:].detach().cpu().numpy()
                self.num_all[i] += 1
         
        tsne = TSNE(3, verbose=1)
        pca = PCA(n_components=3)
        scaler = StandardScaler()
        cmap = cm.get_cmap('tab20')
        # fig, ax = plt.figure()
        ax = plt.subplot(111, projection='3d')

        class_now = [3, 4, 5]
        classnameall = ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle', 'person', 'wind-mill')
        
        for i in class_now:
        # for i in range(self.num_classes):
            if self.num_all[i] == 0:
                continue
            if self.num_all[i] >= 5000:
                feats_now = self.feats_all[i]
            else:
                feats_now = self.feats_all[i][0:self.num_all[i],:]

            # feats_now = scaler.fit_transform(feats_now)
            # tsne_proj = tsne.fit_transform(feats_now)
            tsne_proj = feats_now
            pca.fit(tsne_proj)
            tsne_proj = pca.transform(tsne_proj)

            for j in range(3):
                tsne_proj[:,j] = tsne_proj[:,j] + min(tsne_proj[:,j])
                tsne_proj[:,j] = tsne_proj[:,j] / max(tsne_proj[:,j])
            # row_norms = np.linalg.norm(tsne_proj, axis=1, keepdims=True)
            # tsne_proj = tsne_proj / row_norms

            # import pdb; pdb.set_trace()
            ax.scatter(tsne_proj[:, 0],
                       tsne_proj[:, 1],
                       tsne_proj[:, 2],
                       c=np.array(cmap(i)).reshape(1, 4),
                       label=classnameall[i],
                       alpha=0.5)
        ax.legend(loc='upper left', fontsize='large', markerscale=2)
        plt.savefig("/home/zhr/mmdet-rfla/data/orth_visual/fcos.png")
        plt.close()
        '''

        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'cls_feats'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   cls_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """

        # expand_as这个函数就是把一个tensor变成和函数括号内一样形状的tensor
        # [None]是对应维度增加一维度
        # 这里就是把FPN各个层对应的尺度限制转化一下size方便下面用
        # 一般情况就是五个范围：regress_ranges=
        # ((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))

        '''
        >>> points
tensor([[0., 0.],
        [1., 0.],
        [2., 0.],
        [3., 0.],
        [4., 0.],
        [0., 1.],
        [1., 1.],
        [2., 1.],
        [3., 1.],
        [4., 1.],
        [0., 2.],
        [1., 2.],
        [2., 2.],
        [3., 2.],
        [4., 2.],
        [0., 3.],
        [1., 3.],
        [2., 3.],
        [3., 3.],
        [4., 3.],
        [0., 4.],
        [1., 4.],
        [2., 4.],
        [3., 4.],
        [4., 4.],
        [0., 5.],
        [1., 5.],
        [2., 5.],
        [3., 5.],
        [4., 5.],
        [0., 6.],
        [1., 6.],
        [2., 6.],
        [3., 6.],
        [4., 6.],
        [0., 7.],
        [1., 7.],
        [2., 7.],
        [3., 7.],
        [4., 7.],
        [0., 8.],
        [1., 8.],
        [2., 8.],
        [3., 8.],
        [4., 8.],
        [0., 9.],
        [1., 9.],
        [2., 9.],
        [3., 9.],
        [4., 9.]])
>>> points.new_tensor((-1, 64))[None].expand_as(points)
tensor([[-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.],
        [-1., 64.]])
        '''
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        # 相当于只是生成了一个和原来的points一样大小的（-1， 64）用于对相同个数的anchor进行大小判断，但是实现方式确实很巧妙
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        # concat_points代表把各个level的anchor点按照最高维度拼接一下
        # points的第ｉ个元素的size为（hi*wi，２），拼接完的shape为(所有level的点的数目，2)
        # 之所以合并是为了丢到一个tensor里一起算
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl, 相当于每一个feature_map的points的个数
        # num_points代表每个level里anchor点的数目
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        # 对一个batch里的每个图单独算，跳转到下面看这个函数_get_target_single
        # 可以看到是算的每个图的每个anchor点的分类target和回归target，然后拼成list
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        # >>> a= torch.Tensor([10, 20])
        # >>> a[None]
        # out: tensor([[10., 20.]]), 然后向下扩充num_points次
        areas = areas[None].repeat(num_points, 1)

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        # gt_bboxes的size为（num_gts，4）,加个[None]就是加一维，变为[1,num_gts，4]
        # 继续expand变为指定维度(num_points, num_gts, 4)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        # points之前说了size为（num_points，2),所以xs就是（num_points，）
        xs, ys = points[:, 0], points[:, 1]
        # 所以[:, None]就是（num_points，1）,expand之后就是（num_points，num_gts）
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        #gt_bboxes的size为(num_points, num_gts, 4)，取[..., 0]后变为(num_points, num_gts)
        #所以很明显这里其实就是每个点和每个框之间的上下左右的差值
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys

        # stack完了之后就变为(num_points, num_gts，4)
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # 在使用fcos时这个参数我们设为True
            # 同时有一个redius我们设置为1.5
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            # 计算所有 gt bbox 各自的中心坐标
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            # 此时gt_bboxes的维度为(num_points, num_gts, 4)
            center_gts = torch.zeros_like(gt_bboxes)
            # 此时gt_bboxes的维度为(num_points, num_gts, 1)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            # 中心采样参数是全局参数，需要和各个输出层 stride 联合起来作为控制参数
            # 例如 radius=1.5，对于 stride=4 的层，采样半径是 1.5x4
            # 对于 stride=8 的层，采样半径是 1.5x8
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                # num_points_lvl 代表每一个level中anchor点的个数
                lvl_end = lvl_begin + num_points_lvl
                # strides=[8, 16, 32, 64, 128]
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            # 计算中心扩展范围后新的 gt bbox，实际上相当于向内缩小了 
            # 现在的GT相当于是一个圆，亦或者时
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            # 此时center_gts的维度为(num_points, num_gts, 4)
            # 此时x_mins的维度为(num_points, num_gts, 1)
            '''
            函数原型：torch.where(condition, x, y) → Tensor
            作用： 将两个tensor ： x和y进行逐元素合并，condition ，x和y需要是相同形状的、或者可以广播为相同形状。
            假设合并后的结果为z，z的形状为x，y两者的形状或者他们广播后的形状。
            对于每个下标i，如果condition[i] 满足条件，那么z[i]等于x[i]，否则z[i]等于y[i] 。
            '''

            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            # 计算每个点距 center_gts 4 条边的距离
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            # stack完了之后就变为(num_points, num_gts，4)
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)

            ##################################################
            # 此时直接判断这里面True的个数就可以得到正样本的个数了
            ##################################################
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        # 对于每个点，计算最长边值，bbox_targets shape 为 (num_points, num_gts，4)
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        # 此时由于多个 gt bbox 有交集，需要确定该正样本点到底负责哪个 gt bbox，分配原则是面积最小
        # 先将所有已经被认为是负样本的 areas 全部设置为近似无穷大
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        # areas shape 是 (num_points, num_gts)
        # 选择出每个正样本点所对应 gt bbox 中面积最小的
        min_area, min_area_inds = areas.min(dim=1)

        # 所有样本点的 label，shape 为 (num_points,)
        labels = gt_labels[min_area_inds]
        # 设置背景样本
        labels[min_area == INF] = self.num_classes  # set as BG
        # 所有样本所对应的和 gt bbox 计算出的 targets，shape 为 (num_points, 4)
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
