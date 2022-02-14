import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

norm_cfg = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}
# _norm = 'SyncBN'
_norm = 'BN'
norm_layer = norm_cfg[_norm]

PRIMITIVES = {
  'skip':     lambda input_size, in_channels, out_channels, stride: Identity('skip', input_size, in_channels, out_channels, stride),
  'ir_k3_e1': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 3),
  'ir_k3_e1_r2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 3, dilation=2),
  'ir_k3_e1_r3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 3, dilation=3),
  'ir_k3_e3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 3, stride, 3),
  'ir_k3_e3_r2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 3, stride, 3, dilation=2),
  'ir_k3_e3_r3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 3, stride, 3, dilation=3),
  'ir_k3_e6': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 3),
  'ir_k3_e6_r2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 3, dilation=2),
  'ir_k3_e6_r3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 3, dilation=3),
  'ir_k3_s2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 3, 2),
  'ir_k5_e1': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 5),
  'ir_k5_e1_r2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 5, dilation=2),
  'ir_k5_e1_r3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 5, dilation=3),
  'ir_k5_e3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 3, stride, 5),
  'ir_k5_e6': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 5),
  'ir_k5_e6_r2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 5, dilation=2),
  'ir_k5_e6_r3': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 5, dilation=3),
  'ir_k5_s2': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 1, stride, 5, 2),
  'ir_k7_e6': lambda input_size, in_channels, out_channels, stride: MBBlock(input_size, in_channels, out_channels, 6, stride, 7),
  'sep_k3' : lambda input_size, in_channels, out_channels, stride: SepBlock('sep_k3', input_size, in_channels, out_channels, 1, stride, 3),
  'sep_k5' : lambda input_size, in_channels, out_channels, stride: SepBlock('sep_k5', input_size, in_channels, out_channels, 1, stride, 5),
  'conv1' : lambda input_size, in_channels, out_channels, stride: ConvBNReLU(input_size, in_channels, out_channels, 1, stride, 0, False, 'relu', 'bn'),
  'conv3' : lambda input_size, in_channels, out_channels, stride: ConvBNReLU(input_size, in_channels, out_channels, 3, stride, 1, False, 'relu', 'bn'),
}


class AvgPool(nn.Module):
  def __init__(self, args, input_size, in_channels, stride):
    super(AvgPool, self).__init__()
    self.args, self.stride = args, stride

  def forward(self, x):
    return F.avg_pool2d(x, self.stride)


class ChannelShuffle(nn.Module):
  def __init__(self, input_size, in_channels, groups=1):
    super(ChannelShuffle, self).__init__()
    self.groups = groups

  def forward(self, x):
    if self.groups == 1:
        return x
    N, C, H, W = x.size()
    cpg = C // self.groups # channels per group
    out = x.view(N, self.groups, cpg, H, W)
    out = out.permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(N, C, H, W)
    return out

class ConvBNReLU(nn.Module):
  def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, padding, bias, relu_type, bn_type, groups=1, dilation=1):
    super(ConvBNReLU, self).__init__()
    assert(relu_type in ['relu', 'none'])
    self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias, self.relu_type, self.bn_type, self.groups, self.dilation = \
      in_channels, out_channels, kernel_size, stride, padding, bias, relu_type, bn_type, groups, dilation
    self.input_size = input_size

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
    if self.conv.bias is not None:
      nn.init.constant_(self.conv.bias, 0.0)
    if bn_type == 'bn':
      self.bn = norm_layer(out_channels)
    if bn_type == 'gn':
      self.bn = nn.GroupNorm(gn_group, num_channels = out_channels)
    if relu_type == 'relu':
      self.relu = nn.ReLU(inplace=True)
    else:
      self.relu = nn.Sequential()
    
  def forward(self, x):
    out = self.conv(x)
    out = self.relu(self.bn(out))
    return out


class Identity(nn.Module):
  def __init__(self, genotype, input_size, in_channels, out_channels, stride):
    super(Identity, self).__init__()
    if in_channels != out_channels or stride != 1:
      self.conv = ConvBNReLU(input_size, in_channels, out_channels, kernel_size=1, stride=stride, padding=0, 
                  bias=False, relu_type='relu', bn_type='bn')
    else:
      self.conv = nn.Sequential()

  def forward(self, x):
    if isinstance(self.conv, ConvBNReLU):
      return self.conv(x)
    else:
      return x


class SepBlock(nn.Module):
  def __init__(self, genotype, input_size, in_channels, out_channels, expansion, stride, kernel_size, groups=1, bn_type='bn'):
    super(SepBlock, self).__init__()
    padding = (kernel_size - 1) // 2
    self.input_size, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.expansion, self.groups, self.bn_type, self.padding = \
        input_size, in_channels, out_channels, kernel_size, stride, expansion, groups, bn_type, padding

    self.conv1 = ConvBNReLU(input_size, in_channels, in_channels, kernel_size = kernel_size, stride=self.stride, padding = padding, bias=False, relu_type='relu', bn_type=bn_type, groups = in_channels)
    self.conv2 = ConvBNReLU(input_size // stride, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, relu_type='none', bn_type=bn_type, groups = groups)
 
  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    return out


class MBBlock(nn.Module):
  def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, dilation=1, groups=1, has_se=False, bn_type='bn'):
    super(MBBlock, self).__init__()
    padding = (kernel_size - 1) * dilation // 2
    self.in_channels, self.out_channels, self.kernel_size, self.has_se, self.stride, self.expansion, self.groups, self.bn_type, self.padding, self.dilation = \
        in_channels, out_channels, kernel_size, has_se, stride, expansion, groups, bn_type, padding, dilation
    mid_channels = self.in_channels * self.expansion

    self.conv1 = ConvBNReLU(input_size, in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 bias=False, relu_type='relu', bn_type=bn_type, groups = groups)
    self.conv2 = ConvBNReLU(input_size, mid_channels, mid_channels, kernel_size = kernel_size, stride=self.stride, padding = padding, dilation=dilation,
                 bias=False, relu_type='relu', bn_type=bn_type, groups = mid_channels)
    self.conv3 = ConvBNReLU(input_size//self.stride, mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, 
                 bias=False, relu_type='none', bn_type=bn_type, groups = groups)
   
    self.shuffle = ChannelShuffle(input_size, self.in_channels, self.groups)
 
  def forward(self, x):
    out = self.conv1(x)
    if not self.groups == 1:
      out = self.shuffle(out)
    out = self.conv2(out)
    if self.has_se == True:
      se_out = self.se(out)
      out = out * se_out
    out = self.conv3(out)
    if self.in_channels == self.out_channels and self.stride == 1:
      out = out + x
    return out
