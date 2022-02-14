import numpy as np
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
import torch
import torch.nn as nn

import os.path as osp
latency_lookup_table = {}
# table_file_name = "latency_lookup_table.npy"
# if osp.isfile(table_file_name):
#     latency_lookup_table = np.load(table_file_name).item()

import torch.nn.functional as F
from collections import OrderedDict
from layers import NaiveSyncBatchNorm
from operations import ConvNorm
from att_sa import Self_Attn
BatchNorm2d = NaiveSyncBatchNorm

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1,
                 has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


# For DFN
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2

        return fm


class BNRefine(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1,
                                       ksize // 2, has_bias=has_bias,
                                       norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


# For BiSeNet
class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1, Fch=16, scale=4, branch=2, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        # self.channel_attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
        #                has_bn=False, norm_layer=norm_layer,
        #                has_relu=True, has_bias=False),
        #     ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
        #                has_bn=False, norm_layer=norm_layer,
        #                has_relu=False, has_bias=False),
        #     nn.Sigmoid()
        # )
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    @staticmethod
    def _latency(h, w, C_in, C_out):
        layer = FeatureFusion(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        name = "ff_H%d_W%d_C%d"%(size[1], size[2], size[0])
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, size
        else:
            print("not found in latency_lookup_table:", name)
            latency = FeatureFusion._latency(size[1], size[2], self._scale*self._Fch*self._branch, self._scale*self._Fch*self._branch)
            latency_lookup_table[name] = latency
            np.save("latency_lookup_table.npy", latency_lookup_table)
            return latency, size

    def forward(self, fm):
        # fm is already a concatenation of multiple scales
        fm = self.conv_1x1(fm)
        return fm
        # fm_se = self.channel_attention(fm)
        # output = fm + fm * fm_se
        # return output


class Head(nn.Module):
    def __init__(self, in_planes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d, fmap_size=(128, 256)):
        super(Head, self).__init__()
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        else:
            # in_planes > 256:
            if is_aux:
                mid_planes = in_planes // 2
            else:
                mid_planes = in_planes // 2

        self.att_sa = Self_Attn(dim=in_planes, fmap_size=fmap_size, dim_out=mid_planes, proj_factor=4, downsample=False)
        # self.conv_3x3 = ConvBnRelu(in_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    @staticmethod
    def _latency(h, w, C_in, C_out=19):
        layer = Head(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        assert size[0] == self._in_planes, "size[0] %d, self._in_planes %d"%(size[0], self._in_planes)
        name = "head_H%d_W%d_Cin%d_Cout%d"%(size[1], size[2], size[0], self._out_planes)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, (self._out_planes, size[1], size[2])
        else:
            print("not found in latency_lookup_table:", name)
            latency = Head._latency(size[1], size[2], self._scale*self._Fch*self._branch, self._out_planes)
            latency_lookup_table[name] = latency
            np.save("latency_lookup_table.npy", latency_lookup_table)
            return latency, (self._out_planes, size[1], size[2])

    def forward(self, x):
        # fm = self.conv_3x3(x)
        fm = self.att_sa(x) 
        output = self.conv_1x1(fm)
        return output

class Decoder(nn.Module):
    def __init__(self, in_planes, low_level_inplanes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d, fmap_size=(128, 256)):
        super(Decoder, self).__init__()
        C_low = 48
        self.feature_projection = ConvNorm(low_level_inplanes, C_low, kernel_size=1, stride=1, padding=0, bias=False, groups=1, slimmable=False)
    
        # in_planes = in_planes + C_low
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        else:
            # in_planes > 256:
            if is_aux:
                mid_planes = in_planes // 2
            else:
                mid_planes = in_planes // 2

        
        self.att_sa = Self_Attn(dim=in_planes, fmap_size=fmap_size, dim_out=mid_planes, proj_factor=4, downsample=False)
        self.conv_3x3 = ConvBnRelu(mid_planes + C_low, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    @staticmethod
    def _latency(h, w, C_in, C_out=19):
        layer = Head(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        assert size[0] == self._in_planes, "size[0] %d, self._in_planes %d"%(size[0], self._in_planes)
        name = "head_H%d_W%d_Cin%d_Cout%d"%(size[1], size[2], size[0], self._out_planes)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, (self._out_planes, size[1], size[2])
        else:
            print("not found in latency_lookup_table:", name)
            latency = Head._latency(size[1], size[2], self._scale*self._Fch*self._branch, self._out_planes)
            latency_lookup_table[name] = latency
            np.save("latency_lookup_table.npy", latency_lookup_table)
            return latency, (self._out_planes, size[1], size[2])

    def forward(self, x, low_level_feat):
        low_level_feat = self.feature_projection(low_level_feat)
        x = self.att_sa(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feat), dim=1)
        # x = self.att_sa(x)
        x = self.conv_3x3(x)
        output = self.conv_1x1(x)
        return output

class BasicResidual_downup_2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1

        self.relu = nn.ReLU(inplace=True)
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

class PanopticHead(nn.Module):
    def __init__(self, in_planes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d, fmap_size=(128, 256)):
        super(PanopticHead, self).__init__()
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        else:
            # in_planes > 256:
            if is_aux:
                mid_planes = in_planes // 2
            else:
                mid_planes = in_planes // 2
        
        decoder2_planes = mid_planes // 2

        self.att_sa = Self_Attn(dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, proj_factor=4, downsample=False)
        self.decoder1 = BasicResidual_downup_2x(in_planes, mid_planes, 3, 1, 1)
        self.conv_3x3 = ConvBnRelu(mid_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

        # self.att_sa2 = Self_Attn(dim=in_planes, fmap_size=(128, 256), dim_out=mid_planes, proj_factor=4, downsample=False)
        self.decoder2 = BasicResidual_downup_2x(in_planes, decoder2_planes, 3, 1, 1)
        self.center_conv_3x3 = ConvBnRelu(decoder2_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.center_conv_1x1 = nn.Conv2d(mid_planes, 1, kernel_size=1, stride=1, padding=0)

        self.offset_conv_3x3 = ConvBnRelu(decoder2_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.offset_conv_1x1 = nn.Conv2d(mid_planes, 2, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _latency(h, w, C_in, C_out=19):
        layer = PanopticHead(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        assert size[0] == self._in_planes, "size[0] %d, self._in_planes %d"%(size[0], self._in_planes)
        name = "panoptichead%d_W%d_Cin%d_Cout%d"%(size[1], size[2], size[0], self._out_planes)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, (self._out_planes, size[1], size[2])
        else:
            print("not found in latency_lookup_table:", name)
            latency = Head._latency(size[1], size[2], self._scale*self._Fch*self._branch, self._out_planes)
            latency_lookup_table[name] = latency
            np.save("latency_lookup_table.npy", latency_lookup_table)
            return latency, (self._out_planes, size[1], size[2])

    def forward(self, x):
        output_dict = OrderedDict()
        xs = self.att_sa(x)

        # semantic = self.att_sa1(x)
        semantic = self.decoder1(xs)
        semantic = self.conv_3x3(semantic)
        semantic = self.conv_1x1(semantic)

        # other = self.att_sa2(x)
        other = self.decoder2(x)
        center = self.center_conv_3x3(other)
        center = self.center_conv_1x1(center)

        offset = self.offset_conv_3x3(other)
        offset = self.offset_conv_1x1(offset)

        output_dict['semantic'] = semantic
        output_dict['center'] = center
        output_dict['offset'] = offset
    
        return output_dict

class PanopticHeadDecoder(nn.Module):
    def __init__(self, in_planes, low_level_inplanes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d, fmap_size=(128, 256)):
        super(PanopticHeadDecoder, self).__init__()

        C_low = 48
        self.feature_projection = ConvNorm(low_level_inplanes, C_low, kernel_size=1, stride=1, padding=0, bias=False, groups=1, slimmable=False)
        self.feature_projection_sem = ConvNorm(low_level_inplanes, C_low, kernel_size=1, stride=1, padding=0, bias=False, groups=1, slimmable=False)
        # in_planes = in_planes + C_low

        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        else:
            # in_planes > 256:
            if is_aux:
                mid_planes = in_planes // 2
            else:
                mid_planes = in_planes // 2
        
        decoder2_planes = mid_planes // 2

        self.att_sa = Self_Attn(dim=in_planes, fmap_size=fmap_size, dim_out=in_planes, proj_factor=4, downsample=False)

        # self.att_sa1 = Self_Attn(dim=in_planes, fmap_size=(128, 256), dim_out=mid_planes, proj_factor=4, downsample=False)
        self.decoder1 = BasicResidual_downup_2x(in_planes+C_low, mid_planes, 3, 1, 1)
        self.conv_3x3 = ConvBnRelu(mid_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

        # self.att_sa2 = Self_Attn(dim=in_planes, fmap_size=(128, 256), dim_out=mid_planes, proj_factor=4, downsample=False)
        self.decoder2 = BasicResidual_downup_2x(in_planes+C_low, decoder2_planes, 3, 1, 1)
        self.center_conv_3x3 = ConvBnRelu(decoder2_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.center_conv_1x1 = nn.Conv2d(mid_planes, 1, kernel_size=1, stride=1, padding=0)

        self.offset_conv_3x3 = ConvBnRelu(decoder2_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.offset_conv_1x1 = nn.Conv2d(mid_planes, 2, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _latency(h, w, C_in, C_out=19):
        layer = PanopticHead(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        assert size[0] == self._in_planes, "size[0] %d, self._in_planes %d"%(size[0], self._in_planes)
        name = "panopticheaddecoder%d_W%d_Cin%d_Cout%d"%(size[1], size[2], size[0], self._out_planes)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, (self._out_planes, size[1], size[2])
        else:
            print("not found in latency_lookup_table:", name)
            latency = Head._latency(size[1], size[2], self._scale*self._Fch*self._branch, self._out_planes)
            latency_lookup_table[name] = latency
            np.save("latency_lookup_table.npy", latency_lookup_table)
            return latency, (self._out_planes, size[1], size[2])

    def forward(self, x, low_level_feat):
        output_dict = OrderedDict()
        

        xs = self.att_sa(x)
        low_level_feat_sem = self.feature_projection_sem(low_level_feat)
        xs = F.interpolate(xs, size=low_level_feat_sem.size()[2:], mode='bilinear', align_corners=False)
        xs = torch.cat((xs, low_level_feat_sem), dim=1)

        semantic = self.decoder1(xs)
        semantic = self.conv_3x3(semantic)
        semantic = self.conv_1x1(semantic)

        low_level_feat = self.feature_projection(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feat), dim=1)

        other = self.decoder2(x)
        center = self.center_conv_3x3(other)
        center = self.center_conv_1x1(center)

        offset = self.offset_conv_3x3(other)
        offset = self.offset_conv_1x1(offset)

        output_dict['semantic'] = semantic
        output_dict['center'] = center
        output_dict['offset'] = offset

        return output_dict
