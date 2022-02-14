# ------------------------------------------------------------------------------
# Reference: https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/models/backbones/xception.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['Xception65', 'xception65']


model_urls = {
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
}


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)


class XceptionBlock(nn.Module):
    def __init__(self, channel_list, stride=1, dilation=1, skip_connection_type='conv', relu_first=True,
                 low_feat=False, norm_layer=nn.BatchNorm2d):
        super(XceptionBlock, self).__init__()

        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat

        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1, stride=stride, bias=False)
            self.bn = norm_layer(channel_list[-1])

        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1], dilation=dilation,
                                         relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2], dilation=dilation,
                                         relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3], dilation=dilation,
                                         relu_first=relu_first, stride=stride, norm_layer=norm_layer)
        self.last_inp_channels = channel_list[3]

    def forward(self, inputs):
        sc1 = self.sep_conv1(inputs)
        sc2 = self.sep_conv2(sc1)
        residual = self.sep_conv3(sc2)

        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.bn(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

        if self.low_feat:
            return outputs, sc2
        else:
            return outputs


class Xception65(nn.Module):
    def __init__(self, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Xception65, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        if replace_stride_with_dilation[1]:
            assert replace_stride_with_dilation[2]
            output_stride = 8
        elif replace_stride_with_dilation[2]:
            output_stride = 16
        else:
            output_stride = 32

        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
            exit_block_stride = 2
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
            exit_block_stride = 1
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
            exit_block_stride = 1
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2, norm_layer=norm_layer)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2, low_feat=True, norm_layer=norm_layer)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=entry_block3_stride, low_feat=True,
                                    norm_layer=norm_layer)

        # Middle flow (16 units)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)

        # Exit flow
        self.block20 = XceptionBlock([728, 728, 1024, 1024], stride=exit_block_stride,
                                     dilation=exit_block_dilations[0], norm_layer=norm_layer)
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=exit_block_dilations[1],
                                     skip_connection_type='none', relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        outputs = {}
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        outputs['stem'] = x

        x = self.block1(x)
        x, c1 = self.block2(x)  # b, h//4, w//4, 256
        outputs['res2'] = c1
        x, c2 = self.block3(x)  # b, h//8, w//8, 728
        outputs['res3'] = c2

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        c3 = self.block19(x)
        outputs['res4'] = c3

        # Exit flow
        x = self.block20(c3)
        c4 = self.block21(x)
        outputs['res5'] = c4

        return outputs


def xception65(pretrained=False, progress=True, **kwargs):
    model = Xception65(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['xception65'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
