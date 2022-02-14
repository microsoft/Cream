# --------------------------------------------------------
# Copyright (c) 2019 Jianyuan Guo (guojianyuan1@huawei.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mbblock_ops import OPS

PRIMITIVES = [
    'ir_k3_e3',
    'ir_k3_e6',
    'ir_k3_e6_r2',
    'ir_k5_e3',
    'ir_k5_e6',
    'ir_k7_e6'
]

norm_cfg_ = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}
norm_layer = norm_cfg_['BN']

class MbblockHead(nn.Module):
    def __init__(self, latency=None, gamma=0.02, genotype=None, **kwargs):
        super(MbblockHead, self).__init__()
        self.latency = latency
        self.gamma = gamma
        self.genotype = genotype
        self.last_dim = kwargs.get('out_channels', [256])[-1]
        self.strides = kwargs.get('strides')
        self.out_channels = kwargs.get('out_channels')
        bn_type = kwargs.get('bn_type', 'BN')

        self.cells = nn.ModuleList()
        input_size = 7
        _in_channel = self.last_dim # usually the same as input channel in detector

        for _genotype, _stride, _out_channel in zip(genotype, self.strides, self.out_channels):
            self.cells.append(OPS[_genotype](input_size, _in_channel, _out_channel, _stride, bn=bn_type))
            input_size = input_size // _stride
            _in_channel = _out_channel

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)

        return x, None