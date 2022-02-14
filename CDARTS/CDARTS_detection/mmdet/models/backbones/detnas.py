import logging

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from .utils import load_checkpoint

from ..registry import BACKBONES

norm_cfg = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}
_norm = 'SyncBN'
norm_layer = norm_cfg[_norm]


blocks_key = [
    'shufflenet_3x3',
    'shufflenet_5x5',
    'shufflenet_7x7',
    'xception_3x3',
]


Blocks = {
  'shufflenet_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 3, stride, bn_training),
  'shufflenet_5x5': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 5, stride, bn_training),
  'shufflenet_7x7': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 7, stride, bn_training),
  'xception_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: xception(prefix, in_channels, output_channels, base_mid_channels, stride, bn_training),
}


def create_spatial_conv2d_group_bn_relu(prefix, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1,
                          bias=False, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True,
                          conv_name_fun=None, bn_name_fun=None, bn_training=True, fix_weights=False):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    layer = nn.Sequential()

    if has_spatial_conv:
        spatial_conv_name = conv_name + '_s'
        layer.add_module(spatial_conv_name, nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation, groups=in_channels, bias=bias))
        if fix_weights:
            pass

        if has_spatial_conv_bn:
            layer.add_module(spatial_conv_name + '_bn', norm_layer(in_channels))

    if channel_shuffle:
        pass

    assert in_channels % groups == 0
    assert out_channels % groups == 0

    layer.add_module(conv_name, nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=1, stride=1, padding=0,
                                                    groups=groups, bias=bias))
    if fix_weights:
        pass

    if has_bn:
        bn_name = 'bn_' + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        layer.add_module(bn_name, norm_layer(out_channels))
        if bn_training:
            pass

    if has_relu:
        layer.add_module('relu' + prefix, nn.ReLU(inplace=True))

    return layer


def conv1x1_dwconv_conv1x1(prefix, in_channels, out_channels, mid_channels, kernel_size, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels,
                                                     kernel_size=-1, stride=1, padding=0, groups=1, has_bn=True, has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=False, has_spatial_conv_bn=False,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=1,
                                                     has_bn=True, has_relu=False, channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    return nn.Sequential(*layer)


def xception(prefix, in_channels, out_channels, mid_channels, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels,
                                                     kernel_size=3, stride=stride, padding=1, groups=1, has_bn=True, has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                                                     has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2c', in_channels=mid_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                                                     has_relu=False,
                                                     channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    return nn.Sequential(*layer)


class ConvBNReLU(nn.Module):

    def __init__(self, in_channel, out_channel, k_size, stride=1, padding=0, groups=1,
                 has_bn=True, has_relu=True, gaussian_init=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=True)
        if gaussian_init:
            nn.init.normal_(self.conv.weight.data, 0, 0.01)

        if has_bn:
            self.bn = norm_layer(out_channel)

        self.has_bn = has_bn
        self.has_relu = has_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


def channel_shuffle2(x):
    channels = x.shape[1]
    assert channels % 4 == 0

    height = x.shape[2]
    width = x.shape[3]

    x = x.reshape(x.shape[0] * channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2BlockSearched(nn.Module):
    def __init__(self, prefix, in_channels, out_channels, stride, base_mid_channels, i_th, architecture):
        super(ShuffleNetV2BlockSearched, self).__init__()
        op = blocks_key[architecture[i_th]]
        self.ksize = int(op.split('_')[1][0])
        self.stride = stride
        if self.stride == 2:
            self.conv = Blocks[op](prefix + '_' + op, in_channels, out_channels - in_channels, base_mid_channels, stride, True)
        else:
            self.conv = Blocks[op](prefix + '_' + op, in_channels // 2, out_channels // 2, base_mid_channels, stride, True)
        if stride > 1:
            self.proj_conv = create_spatial_conv2d_group_bn_relu(prefix + '_proj', in_channels, in_channels, self.ksize,
                                                                 stride, self.ksize // 2,
                                                                 has_bn=True, has_relu=True, channel_shuffle=False,
                                                                 has_spatial_conv=True, has_spatial_conv_bn=True,
                                                                 conv_name_fun=lambda p: 'interstellar' + p,
                                                                 bn_name_fun=lambda p: 'bn' + p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_in):
        if self.stride == 1:
            x_proj, x = channel_shuffle2(x_in)
        else:
            x_proj = x_in
            x = x_in
            x_proj = self.proj_conv(x_proj)
        x = self.relu(self.conv(x))

        return torch.cat((x_proj, x), dim=1)


@BACKBONES.register_module
class DetNas(nn.Module):
    def __init__(self, model_size='VOC_FPN_300M', out_indices=(3, 7, 15, 19), frozen_stages=-1):
        super(DetNas, self).__init__()
        print('Model size is {}.'.format(model_size))
        self.out_indices = out_indices
        self.frozen_stages=frozen_stages

        if model_size == 'COCO_FPN_3.8G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 72, 172, 432, 864, 1728, 1728]
        elif model_size == 'COCO_FPN_1.3G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        elif model_size == 'COCO_FPN_300M':
            architecture = [2, 1, 2, 0, 2, 1, 1, 2, 3, 3, 1, 3, 0, 0, 3, 1, 3, 1, 3, 2]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'COCO_RetinaNet_300M':
            architecture = [2, 3, 1, 1, 3, 2, 1, 3, 3, 1, 1, 1, 3, 3, 2, 0, 3, 3, 3, 3]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'VOC_FPN_300M':
            architecture = [2, 1, 0, 3, 1, 3, 0, 3, 2, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif model_size == 'VOC_RetinaNet_300M':
            architecture = [1, 3, 0, 0, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 0, 2, 3, 1, 1]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        else:
            raise NotImplementedError

        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()

        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, i_th=i_th,
                                                               architecture=architecture))
                in_channels = out_channels
                i_th += 1

        self.features = nn.Sequential(*self.features)
        
        if self.out_indices[-1] == len(self.features):
            self.last_conv = ConvBNReLU(in_channel=in_channels, out_channel=stage_out_channels[-1], k_size=1, stride=1, padding=0)

        # self.drop_out = nn.Dropout2d(p=0.2)
        # self.global_pool = nn.AvgPool2d(7)
        self._initialize_weights()

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)
                
        self._freeze_stages()
                
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.first_conv.bn.eval()
            for m in [self.first_conv]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(self.frozen_stages):
            self.features[i].eval()
            for param in self.features[i].parameters():
                param.requires_grad = False

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        x = self.first_conv(x)
        
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.out_indices:
                outs.append(x)
                
        if self.out_indices[-1] == len(self.features):
            x = self.last_conv(x)
            outs.append(x)

        # x = self.last_conv(x)
        # x = self.drop_out(x)
        # x = self.global_pool(x).view(x.size(0), -1)
        return tuple(outs)