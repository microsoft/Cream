import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

norm_cfg_ = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}

OPS = {
    'skip':     lambda input_size, in_channels, out_channels, stride, bn='BN': Identity(input_size, in_channels, out_channels, stride),
    'ir_k3_e1': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, bn=bn),
    'ir_k3_e1_r2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, dilation=2, bn=bn),
    'ir_k3_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 3, bn=bn),
    'ir_k3_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 3, bn=bn),
    'ir_k3_e6_r2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 3, dilation=2, bn=bn),
    'ir_k3_s2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 3, 2, bn=bn),
    'ir_k5_e1': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, bn=bn),
    'ir_k5_e1_r2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, dilation=2, bn=bn),
    'ir_k5_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 5, bn=bn),
    'ir_k5_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 5, bn=bn),
    'ir_k5_e6_r2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 5, dilation=2, bn=bn),
    'ir_k5_s2': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 1, stride, 5, 2, bn=bn),
    'ir_k7_e3': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 3, stride, 7, bn=bn),
    'ir_k7_e6': lambda input_size, in_channels, out_channels, stride, bn='BN': MBBlock(input_size, in_channels, out_channels, 6, stride, 7, bn=bn),
    'sep_k3' : lambda input_size, in_channels, out_channels, stride, bn='BN': SepConv(input_size, in_channels, out_channels, 1, stride, 3),
    'sep_k5' : lambda input_size, in_channels, out_channels, stride, bn='BN': SepConv(input_size, in_channels, out_channels, 1, stride, 5),
    'conv1' : lambda input_size, in_channels, out_channels, stride, bn='BN': ConvBNReLU(input_size, in_channels, out_channels, 1, stride, bn_type=bn),
    'conv3' : lambda input_size, in_channels, out_channels, stride, bn='BN': ConvBNReLU(input_size, in_channels, out_channels, 3, stride, bn_type=bn),
    'conv5' : lambda input_size, in_channels, out_channels, stride, bn='BN': ConvBNReLU(input_size, in_channels, out_channels, 5, stride, bn_type=bn),
    'avgpool': lambda input_size, in_channels, out_channels, stride, bn='BN': AvgPool(input_size, in_channels, stride),
}


class AvgPool(nn.Module):
    def __init__(self, stride):
        super(AvgPool, self).__init__()
        self.stride = stride

    def forward(self, x):
        return F.avg_pool2d(x, self.stride)


class ChannelShuffle(nn.Module):
    def __init__(self, groups=1):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x
        N, C, H, W = x.size()
        cpg = C // self.groups  # channels per group
        out = x.view(N, self.groups, cpg, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(N, C, H, W)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, dilation=1, bias=False, relu_type='relu', bn_type='BN', groups=1):
        super(ConvBNReLU, self).__init__()
        assert(relu_type in ['relu', 'none'])
        padding = (kernel_size - 1) * dilation // 2

        if bn_type == 'none':
            bias = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

        if bn_type == 'none' :
            self.bn = nn.Sequential()
        elif bn_type == 'GN':
            norm_layer = norm_cfg_[bn_type]
            self.bn = norm_layer(num_channels=out_channels, num_groups=32)
        else:
            norm_layer = norm_cfg_[bn_type]
            self.bn = norm_layer(out_channels)

        self.relu = nn.ReLU(inplace=True) if relu_type == 'relu' else nn.Sequential()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(self.bn(out))
        return out

  
class SE(nn.Module):
  def __init__(self, input_size, in_channels, se_ratio):
    super(SE, self).__init__()
    self.in_channels, self.se_ratio = in_channels, se_ratio
    self.pooling = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Conv2d(in_channels, max(1, int(in_channels * se_ratio)), 1, bias=False)
    self.fc2 = nn.Conv2d(max(1, int(in_channels * se_ratio)), in_channels, 1, bias=False)

  def forward(self, x):
    out = self.pooling(x)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    out = F.sigmoid(out)
    return out


class Identity(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.conv = ConvBNReLU(input_size, in_channels, out_channels, kernel_size=1, stride=stride,
                padding=0, bias=False, relu_type='relu', bn_type='bn')
        else:
            self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class SepConv(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, groups=1, bn_type='BN'):
        super(SepConv, self).__init__()
        self.conv1 = ConvBNReLU(input_size, in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
            bias=False, relu_type='relu', bn_type=bn_type, groups=in_channels)
        self.conv2 = ConvBNReLU(input_size//stride, in_channels, out_channels, kernel_size=1, stride=1, 
            bias=False, relu_type='none', bn_type=bn_type, groups=groups)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class MBBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, expansion, stride, kernel_size, dilation=1, groups=1, has_se=False, bn='BN'):
        super(MBBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels =out_channels 
        self.has_se = has_se
        self.stride = stride
        self.groups = groups
        mid_channels = in_channels * expansion

        self.conv1 = ConvBNReLU(input_size, in_channels, mid_channels, kernel_size=1, stride=1, dilation=1,
            bias=False, relu_type='relu', bn_type=bn, groups=groups)
        self.conv2 = ConvBNReLU(input_size, mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
            bias=False, relu_type='relu', bn_type=bn, groups=mid_channels)
        self.conv3 = ConvBNReLU(input_size//self.stride, mid_channels, out_channels, kernel_size=1, stride=1, dilation=1, 
            bias=False, relu_type='none', bn_type=bn, groups=groups)

        if has_se == True:
            self.se = SE(input_size, mid_channels, se_ratio=0.05)
        
        if groups != 1:
            self.shuffle = ChannelShuffle(input_size, in_channels, groups)
 
    def forward(self, x):
        out = self.conv1(x)
        if self.groups != 1:
            out = self.shuffle(out)
        out = self.conv2(out)
        if self.has_se:
            out = out * self.se(out)
        out = self.conv3(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out