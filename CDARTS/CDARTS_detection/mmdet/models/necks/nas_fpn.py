import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn import caffe2_xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule

norm_cfg_ = {
    'BN': nn.BatchNorm2d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
}


class MergingCell(nn.Module):
    def __init__(self, channels=256, with_conv=True, norm_type='BN'):
        super(MergingCell, self).__init__()
        self.with_conv = with_conv
        norm_layer = norm_cfg_[norm_type]
        if self.with_conv:
            self.conv_out = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1),
                norm_layer(channels)
            )

    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode='nearest')
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = x.shape[-1] // size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            # x = F.interpolate(x, size=size, mode='nearest')
            return x

    def forward(self, x1, x2, out_size):
        assert x1.shape[:2] == x2.shape[:2]
        assert len(out_size) == 2

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_conv:
            x = self.conv_out(x)
        return x


class SumCell(MergingCell):
    def _binary_op(self, x1, x2):
        return x1 + x2


class GPCell(MergingCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1


@NECKS.register_module
class NASFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 stack_times=7,
                 lateral_kernel=1,
                 norm_type='SyncBN'):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack_times = stack_times
        self.norm_type = norm_type

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level):  # RetinaNet (1,4)
        for i in range(self.start_level, self.start_level + num_outs):
            in_channel = in_channels[i] if i < self.backbone_end_level else in_channels[-1]
            padding = (lateral_kernel - 1) // 2
            l_conv = nn.Conv2d(in_channel, out_channels, kernel_size=lateral_kernel, padding=padding)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            if self.add_extra_convs:
                extra_conv = nn.Conv2d(in_channels[-1], in_channels[-1], 3, stride=2, padding=1)
                self.extra_downsamples.append(extra_conv)
            else:
                self.extra_downsamples.append(nn.MaxPool2d(2, stride=2))

        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GPCell(out_channels, norm_type=norm_type)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(out_channels, norm_type=norm_type)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(out_channels, norm_type=norm_type)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(out_channels, norm_type=norm_type)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GPCell(with_conv=False)
            stage['sum_55_5'] = SumCell(out_channels, norm_type=norm_type)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GPCell(with_conv=False)
            stage['sum_77_7'] = SumCell(out_channels, norm_type=norm_type)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GPCell(out_channels, norm_type=norm_type)
            self.fpn_stages.append(stage)

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build P6-P7 on top of P5
        inputs = list(inputs)
        for downsample in self.extra_downsamples:
            inputs.append(downsample(inputs[-1]))
            
        # 1x1 on P3-P7
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return tuple([p3, p4, p5, p6, p7])
        