# --------------------------------------------------------
# Copyright (c) 2019 Jianyuan Guo (guojianyuan1@huawei.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hit_ops import OPS

PRIMITIVES = [
    'conv_1x1',
    'ir_k3_e6_d3',
    'ir_k5_e6',
    'ir_k5_e6_d3',
    'sd_k3_d1',
    'sd_k3_d3',
    'sd_k5_d2',
    'sd_k5_d3',
]


class HitNeck(nn.Module):
    def __init__(self, num_fm=4, in_channel=[256], out_channel=256,
                 latency=None, gamma=0.02, genotype=None, **kwargs):
        super(HitNeck, self).__init__()
        self.num_fm = num_fm
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.genotype = genotype
        bn_type = kwargs.get('bn_type', 'BN')

        self.cells = nn.ModuleList()
        input_size = [160, 80, 40, 20]  # 1/4, 1/8, 1/16, 1/32

        for i, ops in enumerate(genotype):
            if i < self.num_fm:
                cell = OPS[PRIMITIVES[ops]](input_size[i%self.num_fm], 
                    in_channel[i%self.num_fm], out_channel, 1, bn=bn_type)
            else:
                cell = OPS[PRIMITIVES[ops]](input_size[i%self.num_fm], 
                    out_channel, out_channel, 1, bn=bn_type)
            self.cells.append(cell)

            
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x, step):
        assert(step in [1, 2])
        _step = step - 1
        out = []

        for i in range(_step*self.num_fm, step*self.num_fm):
            out.append(self.cells[i](x[i%self.num_fm]))         

        return out
