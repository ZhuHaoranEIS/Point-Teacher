# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS as NECKS


@NECKS.register_module()
class PSAGG(BaseModule):
    r"""Feature Pyramid Network.
    """

    def __init__(self,
                 num_aggregation,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PSAGG, self).__init__(init_cfg)
        self.num_aggregation = num_aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_cfg = upsample_cfg.copy()

        self.lateral_convs = nn.ModuleList()
        for i in range(num_aggregation):
            if i != num_aggregation-1:
                l_conv = ConvModule(
                    in_channels,
                    in_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            else:
                l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            self.lateral_convs.append(l_conv)
        
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        inputs = list(inputs)
        ### output aggregation
        inputs[-1] = self.lateral_convs[0](inputs[-1])
        for i in range(self.num_aggregation):
            index = self.num_aggregation - i - 1
            if index != 0:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    inputs[index-1] = inputs[index-1] + F.interpolate(
                        inputs[index], **self.upsample_cfg)
                else:
                    prev_shape = inputs[index-1].shape[2:]
                    inputs[index-1] = inputs[index-1] + F.interpolate(
                        inputs[index], size=prev_shape, **self.upsample_cfg)
                inputs[index-1] = self.lateral_convs[i+1](inputs[index-1])

        outputs = tuple([inputs[0]])
        return outputs
