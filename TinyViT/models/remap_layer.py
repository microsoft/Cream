# --------------------------------------------------------
# TinyViT Utils
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------

import torch
from torch import nn


class RemapLayer(nn.Module):
    def __init__(self, fname):
        super().__init__()
        with open(fname) as fin:
            self.mapping = torch.Tensor(
                list(map(int, fin.readlines()))).to(torch.long)

    def forward(self, x):
        '''
        x: [batch_size, class]
        '''
        B = len(x)
        dummy_cls = x.new_zeros((B, 1))
        expand_x = torch.cat([x, dummy_cls], dim=1)
        return expand_x[:, self.mapping]
