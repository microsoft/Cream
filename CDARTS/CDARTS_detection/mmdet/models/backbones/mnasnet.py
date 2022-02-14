import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from .dropblock import DropBlockScheduled, DropBlock2D

import logging
from torch.nn.modules.batchnorm import _BatchNorm

import torch.nn.functional as F
import time
import numpy as np

from ..registry import BACKBONES


def Conv_3x3(inp, oup, stride, activation=nn.ReLU6, act_params={"inplace": True}):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        activation(**act_params)
    )


def Conv_1x1(inp, oup, activation=nn.ReLU6, act_params={"inplace": True}):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation(**act_params)
    )


def SepConv_3x3(inp, oup, activation=nn.ReLU6, act_params={"inplace": True}):  # input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        activation(**act_params),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel, drop_prob=0.0, num_steps=3e5, activation=nn.ReLU6,
                 act_params={"inplace": True}):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            DropBlockScheduled(
                DropBlock2D(drop_prob=drop_prob, block_size=7),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=num_steps),
            activation(**act_params),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            DropBlockScheduled(
                DropBlock2D(drop_prob=drop_prob, block_size=7),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=num_steps),
            activation(**act_params),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            DropBlockScheduled(
                DropBlock2D(drop_prob=drop_prob, block_size=7),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=num_steps),
        )
        if self.use_res_connect:
            self.skip_drop = DropBlockScheduled(
                DropBlock2D(drop_prob=drop_prob, block_size=7),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=num_steps)
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
        if self.use_res_connect:
            return self.skip_drop(x + self.conv(x))
        else:
            return self.conv(x)


@BACKBONES.register_module
class MnasNet(nn.Module):
    def __init__(self, out_indices=(1,2,3,4), width_mult=1., drop_prob=0.0, num_steps=3e5, activation=nn.ReLU6,
                 act_params={"inplace": True}):
        super(MnasNet, self).__init__()
        self.out_indices = out_indices

        self.activation = activation
        self.act_params = act_params

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k, dp
            [3, 24, 3, 2, 3, 0],  # -> 56x56
            [3, 40, 3, 2, 5, 0],  # -> 28x28
            [6, 80, 3, 2, 5, 0],  # -> 14x14
            [6, 96, 2, 1, 3, drop_prob],  # -> 14x14
            [6, 192, 4, 2, 5, drop_prob],  # -> 7x7
            [6, 320, 1, 1, 3, drop_prob],  # -> 7x7
        ]
        self.num_steps = num_steps

        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.features = [Conv_3x3(3, input_channel, 2, self.activation, self.act_params),
                         SepConv_3x3(input_channel, 16, self.activation, self.act_params)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k, dp in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k, dp, self.num_steps,
                                                          self.activation, self.act_params))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k, dp, self.num_steps,
                                                          self.activation, self.act_params))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel, self.activation, self.act_params))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        import pdb
        pdb.set_trace()
        outs = []
        cnt = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = MnasNet()
    print(net)
    x_image = Variable(torch.randn(1, 3, 224, 224))
    y = net(x_image)
    # print(y)