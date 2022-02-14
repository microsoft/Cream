import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from .utils import load_checkpoint

from ..registry import BACKBONES

norm_cfg = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}
_norm = 'BN'
norm_layer = norm_cfg[_norm]


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, rf_series=1, rf_sd=1, rf_bn=True, rf_relu=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 1, groups=hidden_dim, bias=False),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )
        else:
            self.conv = []
            # pw
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(norm_layer(hidden_dim))
            self.conv.append(nn.ReLU6(inplace=True))
            # dw

            for idx in range(rf_series):
                self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                                           padding=int((kernel_size-1)*(idx+1)/2), 
                                           dilation=idx+1, groups=hidden_dim, bias=False))
                if rf_bn:
                    self.conv.append(norm_layer(hidden_dim))
                if rf_relu:
                    self.conv.append(nn.ReLU6(inplace=True))
            if not rf_bn:
                self.conv.append(norm_layer(hidden_dim))
            if not rf_relu:
                self.conv.append(nn.ReLU6(inplace=True))

            # pw-linear
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(norm_layer(oup))
            self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):

    def __init__(self,
                 width_mult=1.,
                 input_channel=32,
                 last_channel = 1280,
                 kernel_size=3,

                 out_indices=(2, 5, 12, 17),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):

        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 112x112 0
            [6, 24, 2, 2],  # 56x56  2
            [6, 32, 3, 2],  # 28x28  5
            [6, 64, 4, 2],  # 14x14  9
            [6, 96, 3, 1],  # 14x14  12
            [6, 160, 3, 2],  # 7x7  15
            [6, 320, 1, 1],  # 7x7  16
        ]
        self.kernel_size=kernel_size
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.mv2_layer = []
        features = []
        features.append(
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
                norm_layer(input_channel),
                nn.ReLU6(inplace=True)
            )
        )

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t, 
                                               kernel_size=3))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t, 
                                               kernel_size=kernel_size))
                input_channel = output_channel

        features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                norm_layer(last_channel),
                nn.ReLU6(inplace=True)
            )
        )
        for i, module in enumerate(features):
            layer_name = 'features{}'.format(i)
            self.add_module(layer_name, module)
            self.mv2_layer.append(layer_name)

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

        self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.mv2_layer):
            layer = getattr(self, layer_name)
            x = layer(x)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    '''
    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    '''