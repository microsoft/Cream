import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import time
import numpy as np

from .fbnet_blocks import *
from .fbnet_arch import predefine_archs

import logging
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from .utils import load_checkpoint

from ..registry import BACKBONES


@BACKBONES.register_module
class FBNet(nn.Module):
    def __init__(self, arch='fbnet_c', out_indices=(5, 9, 17, 22), frozen_stages=-1):
        super(FBNet, self).__init__()
        print('Model is {}.'.format(arch))
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.arch = arch
        self.input_size = 800

        self.build_backbone(self.arch, self.input_size)

    def build_backbone(self, arch, input_size):
        genotypes = predefine_archs[arch]['genotypes'] 
        strides = predefine_archs[arch]['strides'] 
        out_channels = predefine_archs[arch]['out_channels']
        
        self.layers = nn.ModuleList()
        self.layers.append(ConvBNReLU(input_size, in_channels=3, out_channels=out_channels[0], kernel_size=3, stride=strides[0], padding=1, 
                      bias=True, relu_type='relu', bn_type='bn'))
        input_size = input_size // strides[0]

        _in_channels = out_channels[0]
        for genotype, stride, _out_channels in zip(genotypes[1:], strides[1:], out_channels[1:]):
            if genotype.endswith('sb'):
                self.layers.append(SUPER_PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            else:
                self.layers.append(PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            input_size = input_size // stride
            _in_channels = _out_channels

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

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

    def forward(self, x, alphas=None):
        outs = []
        cnt = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs