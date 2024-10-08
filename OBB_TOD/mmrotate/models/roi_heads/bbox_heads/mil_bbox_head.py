import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import torch.nn.functional as F


@HEADS.register_module()
class Shared2FCInstanceMILHead(BBoxHead):

    def __init__(self,
                 num_stages=1,
                 num_shared_fcs=2,
                 num_cls_fcs=0,
                 num_reg_fcs=0,
                 num_ref_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 loss_type=None,
                 with_loss_pseudo=True,
                 loss_mil1=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25),  # weight
                 loss_mil2=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='fc_cls',
                         std=0.01,
                         bias_prob=0.01), ),
                 *args,
                 **kwargs):
        super(Shared2FCInstanceMILHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_fcs + num_cls_fcs + num_reg_fcs > 0)

        self.with_reg = False ##################################


        if not self.with_cls:
            assert num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_fcs == 0
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.num_ref_fcs = num_ref_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mil1 = build_loss(loss_mil1)
        self.loss_mil2 = build_loss(loss_mil2)
        self.with_ins = self.with_cls
        self.num_stages = num_stages
        self.loss_type = loss_type
        self.with_loss_pseudo = with_loss_pseudo

        # add shared convs and fcs
        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        _, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(0, self.num_cls_fcs, self.shared_out_channels)

        # add ins specific branch
        _, self.ins_fcs, self.ins_last_dim = \
            self._add_conv_fc_branch(0, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        if self. with_reg:
            _, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(0, self.num_reg_fcs, self.shared_out_channels)
        if self.num_stages == 2:
            _, self.ref_fcs, self.ref_last_dim = \
                self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)
            if self.num_stages == 3:
                _, self.ref2_fcs, self.ref2_last_dim = \
                    self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_cls_fcs == 0:
                self.ins_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed

        self.fc_cls = nn.ModuleList()
        self.fc_ins = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        import pdb; pdb.set_trace()
        for i in range(1, self.num_stages):
            if i < 1:
                num_cls = self.num_classes + 1
            else:
                num_cls = self.num_classes
            if self.with_cls:
                self.fc_cls.append(build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.cls_last_dim,
                    out_features=num_cls))
            if self.with_ins:
                self.fc_ins.append(build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.ins_last_dim,
                    out_features=num_cls))
            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                                 self.num_classes)
                self.fc_reg.append(build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=self.reg_last_dim,
                    out_features=out_dim_reg))

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, stage):
        # shared part


        if stage == 0:
            if self.num_shared_fcs > 0:
                if self.with_avg_pool:
                    x = self.avg_pool(x)

                x = x.flatten(1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))

                x_cls = x
                x_ins = x
                x_reg = x


        elif stage >= 1:
            if self.num_ref_fcs > 0:
                x_ref = x
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x_ref = x_ref.flatten(1)
                for fc in self.ref_fcs:
                    x_ref = self.relu(fc(x_ref))
                x_cls = x_ref
                x_ins = x_ref
                x_reg = x_ref
            else:
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x = x.flatten(1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))

                x_cls = x
                x_ins = x
                x_reg = x

        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for fc in self.ins_fcs:
            x_ins = self.relu(fc(x_ins))
        if self.with_reg:
            for fc in self.reg_fcs:
                if x_reg.dim() > 2:
                    x_reg = x_reg.flatten(1)
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls[stage-1](x_cls) if self.with_cls else None
        ins_score = self.fc_ins[stage-1](x_ins) if self.with_ins else None
        reg_box = self.fc_reg[stage-1](x_reg) if self.with_reg else None

        return cls_score, ins_score, reg_box

    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_mil(self,
                 stage,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 reg_box,
                 labels,
                 gt_boxes,
                 label_weights,
                 retrain_weights,
                 reduction_override=None):
        losses = dict()
        from mmdet.models.losses.utils import weight_reduce_loss
        if cls_score is not None:
            if cls_score.numel() > 0:
                label_valid = proposals_valid_list
                cls_score = cls_score.sigmoid()
                num_sample = cls_score.shape[0]
                pos_loss, bag_acc, num_pos = self.loss_mil2(
                    cls_score,
                    ins_score,
                    labels,
                    label_valid,
                    label_weights.unsqueeze(-1), )
                if isinstance(pos_loss, dict):
                    losses.update(pos_loss)
                else:
                    losses['loss_instance_mil'] = pos_loss
                losses['bag_acc'] = bag_acc

        if neg_cls_score is not None:
            num_neg, num_class = neg_cls_score.shape
            neg_cls_score = neg_cls_score.sigmoid()
            neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
            loss_weights = 0.75
            neg_valid = neg_weights.reshape(num_neg, -1)
            assert num_sample != 0
            neg_loss = self.loss_mil2.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
            neg_loss = loss_weights * label_weights.float().mean() * weight_reduce_loss(neg_loss, None,
                                                                                        avg_factor=num_sample)
            losses.update({"neg_loss": neg_loss})
        # #

        return losses
