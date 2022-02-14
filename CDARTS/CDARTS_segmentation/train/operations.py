__all__ = ['ConvNorm', 'BasicResidual1x', 'BasicResidual_downup_1x', 'BasicResidual2x', 'BasicResidual_downup_2x', 'FactorizedReduce', 'OPS', 'OPS_name', 'OPS_Class', 'Self_Attn']

from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict
from torch import nn, einsum
from einops import rearrange

C = edict()
"""please config ROOT_dir and user when u first using"""
# C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.root_dir = osp.realpath("..")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
# C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
from slimmable_ops import USConv2d, USBatchNorm2d
from layers import NaiveSyncBatchNorm


latency_lookup_table = {}
# table_file_name = "latency_lookup_table.npy"
# if osp.isfile(table_file_name):
#     latency_lookup_table = np.load(table_file_name).item()


# BatchNorm2d = nn.BatchNorm2d
BatchNorm2d = NaiveSyncBatchNorm

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x = torch.mul(x, mask)
        # x.div_(keep_prob).mul_(mask)
    return x

class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        latency = 0
        return latency, (c_in, h_in, w_in)

class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = nn.Sequential(
                USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list),
                USBatchNorm2d(C_out, width_mult_list),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias),
                # nn.BatchNorm2d(C_out),
                BatchNorm2d(C_out),
                nn.ReLU(inplace=True),
            )
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv[0].set_ratio(ratio)
        self.conv[1].set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        assert x.size()[1] == self.C_in, "{} {}".format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class BasicResidual1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            if self.stride==1:
                self.downsample = nn.Sequential(
                    USConv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list),
                    USBatchNorm2d(C_out, width_mult_list)
                )
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            if self.stride==1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False),
                    BatchNorm2d(C_out)
                )
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual_downup_1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=False)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=False)
            out = out + self.downsample(x)
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn2 = nn.BatchNorm2d(C_out)
            self.bn2 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
            if self.stride==1:
                self.downsample = nn.Sequential(
                    USConv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list),
                    USBatchNorm2d(C_out, width_mult_list)
                )
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn2 = nn.BatchNorm2d(C_out)
            self.bn2 = BatchNorm2d(C_out)
            if self.stride==1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(C_in, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False),
                    BatchNorm2d(C_out)
                )
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=False)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=False)
            out = out + self.downsample(x)
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        if stride == 1 and slimmable:
            self.conv1 = USConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        elif stride == 2:
            self.relu = nn.ReLU(inplace=True)
            if slimmable:
                self.conv1 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.conv2 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.bn = USBatchNorm2d(C_out, width_mult_list)
            else:
                self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.bn = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        if self.stride == 1:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])
        elif self.stride == 2:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.conv2.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = FactorizedReduce._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            if self.slimmable:
                out = self.conv1(x)
                out = self.bn(out)
                out = self.relu(out)
                return out
            else:
                return x


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads

        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

class Self_Attn(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        slimmable=True,
        width_mult_list=[1.],
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU(inplace=True)
    ):
        super().__init__()

        # shortcut

        # contraction and expansion
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list


        if slimmable:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.sk = False
            self.shortcut = nn.Sequential(
                USConv2d(dim, dim_out, kernel_size, padding=padding, stride=stride, dilation=1, groups=1, bias=False, width_mult_list=width_mult_list),
                USBatchNorm2d(dim_out, width_mult_list),
                activation
            )
        else:
            if dim != dim_out or downsample:
                self.sk = False
                kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

                self.shortcut = nn.Sequential(
                    nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                    BatchNorm2d(dim_out),
                    activation
                )
            else:
                self.sk = True
                self.shortcut = nn.Identity()

        self.mix_bn1 = nn.ModuleList([])
        self.mix_bn2 = nn.ModuleList([])
        self.mix_bn3 = nn.ModuleList([])

        # attn_dim_in = dim_out // proj_factor
        attn_dim_in = dim_out
        # attn_dim_out = heads * dim_head
        attn_dim_out = attn_dim_in

        if self.slimmable:
            self.mix_bn1.append(USBatchNorm2d(dim_out, width_mult_list))
            self.mix_bn2.append(USBatchNorm2d(dim_out, width_mult_list))
            self.mix_bn3.append(USBatchNorm2d(dim_out, width_mult_list))
            nn.init.zeros_(self.mix_bn3[0].weight)
        else:
            self.mix_bn1.append(BatchNorm2d(dim_out))
            self.mix_bn2.append(BatchNorm2d(dim_out))
            self.mix_bn3.append(BatchNorm2d(dim_out))
            nn.init.zeros_(self.mix_bn3[0].weight)

        if self.slimmable:
            self.net1 = USConv2d(dim, attn_dim_in, 1, padding=0, stride=1, dilation=1, groups=1, bias=False, width_mult_list=width_mult_list)

            self.net2 = nn.Sequential(
                activation,
                ATT(attn_dim_in, slimmable=True, width_mult_list=width_mult_list),
                nn.AvgPool2d((2, 2)) if downsample else nn.Identity()
            )

            self.net3 = nn.Sequential(
                activation,
                USConv2d(attn_dim_out, dim_out, 1, padding=0, stride=1, dilation=1, groups=1, bias=False, width_mult_list=width_mult_list),
            )

        else:
            self.net1 = nn.Conv2d(dim, attn_dim_in, 1, bias = False)

            self.net2 = nn.Sequential(
                activation,
                ATT(attn_dim_in, slimmable=False),
                nn.AvgPool2d((2, 2)) if downsample else nn.Identity()
            )

            self.net3 = nn.Sequential(
                activation,
                nn.Conv2d(attn_dim_out, dim_out, 1, bias = False),
            )

        # init last batch norm gamma to zero

        # nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation
    
    def set_ratio(self, ratio):
        if not self.sk:
            self.shortcut[0].set_ratio(ratio)
            self.shortcut[1].set_ratio(ratio[1])

        for i in range(len(self.mix_bn1)):
            self.mix_bn1[i].set_ratio(ratio[1])
            self.mix_bn2[i].set_ratio(ratio[1])
            self.mix_bn3[i].set_ratio(ratio[1])

        self.net1.set_ratio(ratio)
        self.net2[1].set_ratio((ratio[1], ratio[1]))
        self.net3[1].set_ratio((ratio[1], ratio[1]))

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = Self_Attn(dim=C_in, fmap_size=(128, 256), dim_out=C_out, downsample=(stride==2))
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = Self_Attn(dim=C_in, fmap_size=(128, 256), dim_out=C_out, downsample=(stride==2))
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Self_Attn_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = Self_Attn._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        branch = 0
        shortcut = self.shortcut(x)
        x = self.net1(x)
        x = self.mix_bn1[branch](x)
        x = self.net2(x)
        x = self.mix_bn2[branch](x)
        x = self.net3(x)
        x = self.mix_bn3[branch](x)
        x += shortcut
        return self.activation(x)

class ATT(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, slimmable=True, width_mult_list=[1.]):
        super(ATT, self).__init__()
        self.chanel_in = in_dim

        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        
        if self.slimmable:
            self.query_conv = USConv2d(in_dim , in_dim//8 , 1, padding=0, stride=1, bias=False, width_mult_list=width_mult_list)
            self.key_conv = USConv2d(in_dim , in_dim//8 , 1, padding=0, stride=1, bias=False, width_mult_list=width_mult_list)
            self.value_conv = USConv2d(in_dim , in_dim , 1, padding=0, stride=1, bias=False, width_mult_list=width_mult_list)
        else:
            self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
            self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
            self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.query_conv.set_ratio(ratio)
        self.key_conv.set_ratio(ratio)
        self.value_conv.set_ratio(ratio)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1) )
        out = out.view(m_batchsize, C, width,height)
        
        out = self.gamma*out + x
        return out

from collections import OrderedDict
OPS = {
    'skip' : lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: FactorizedReduce(C_in, C_out, stride, slimmable, width_mult_list),
    'conv' : lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: BasicResidual1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: BasicResidual_downup_1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x' : lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: BasicResidual2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: BasicResidual_downup_2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'sa': lambda C_in, C_out, stride, slimmable, width_mult_list, fmap_size: Self_Attn(dim=C_in, fmap_size=(128, 256), dim_out=C_out, proj_factor=1, downsample=(stride==2), slimmable=slimmable, width_mult_list=width_mult_list)
}

OPS_name = ["FactorizedReduce", "BasicResidual1x", "BasicResidual_downup_1x", "BasicResidual2x", "BasicResidual_downup_2x", "Self_Attn"]

OPS_Class = OrderedDict()
OPS_Class['skip'] = FactorizedReduce
OPS_Class['conv'] = BasicResidual1x
OPS_Class['conv_downup'] = BasicResidual_downup_1x
OPS_Class['conv_2x'] = BasicResidual2x
OPS_Class['conv_2x_downup'] = BasicResidual_downup_2x
OPS_Class['sa'] = Self_Attn
