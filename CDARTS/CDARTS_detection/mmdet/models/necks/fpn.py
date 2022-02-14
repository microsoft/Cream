import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


# For toy experiments
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size, dilation=1, groups=1):
        super(MBBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels =out_channels 
        self.stride = stride
        self.groups = groups
        mid_channels = in_channels * expansion
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, groups=mid_channels),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups),
            nn.SyncBatchNorm(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)

            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 fpn_kernel=3,
                 lateral_kernel=1,
                 depthwise=None,
                 toy_replace=None,
                 dense_add=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.fpn_kernel = fpn_kernel
        self.lateral_kernel = lateral_kernel
        self.dense_add = dense_add

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
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                lateral_kernel,
                padding=(lateral_kernel-1)//2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            if depthwise is not None:
                if depthwise == 'sep':
                    fpn_conv = nn.Conv2d(out_channels, out_channels, self.fpn_kernel, 
                        padding=int((self.fpn_kernel-1)/2), groups=out_channels)
                elif depthwise == 'sep-depth':
                    fpn_conv = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, self.fpn_kernel, 
                            padding=int((self.fpn_kernel-1)/2), groups=out_channels),
                        nn.Conv2d(out_channels, out_channels, 1, padding=0))
            else:
                if toy_replace is not None and i == toy_replace.get('stage', 30):
                    if toy_replace.get('block', 'res') == 'ir':
                        fpn_conv = MBBlock(
                            out_channels, out_channels, 1, 1, 
                            toy_replace.get('conv_kernel'), dilation=toy_replace.get('dilation'), groups=1)
                    else:
                        fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        toy_replace.get('conv_kernel'),
                        padding=(toy_replace.get('conv_kernel')-1) * toy_replace.get('dilation') // 2,
                        dilation=toy_replace.get('dilation'),
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)
                else:
                    fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        self.fpn_kernel,
                        padding=int((self.fpn_kernel-1)/2),
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)                
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        if self.dense_add is not None:
            if self.dense_add == 'no':
                laterals = laterals
            elif self.dense_add == 'all':
                laterals_ = [0 for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(len(laterals)):
                        for k in range(i-j):
                            if k == 0:
                                tmp_lateral = F.max_pool2d(laterals[j], 2, stride=2)
                            else:
                                tmp_lateral = F.max_pool2d(tmp_lateral, 2, stride=2)
                        if i > j:
                            laterals_[i] += F.interpolate(tmp_lateral, size=(h,w), mode='bilinear', align_corners=True)
                        else:
                            laterals_[i] += F.interpolate(laterals[j], size=(h,w), mode='bilinear', align_corners=True)
                laterals = laterals_
            elif self.dense_add == 'top-down':
                laterals_ = [0 for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(used_backbone_levels - 1, i-1, -1):
                        laterals_[i] += F.interpolate(laterals[j], size=(h,w), mode='nearest')
                laterals = laterals_
            elif self.dense_add == 'bottom-up-nearest':
                for i in range(0, used_backbone_levels-1, 1):
                    laterals[i+1] += F.max_pool2d(laterals[i], 1, stride=2)
            elif self.dense_add == 'bottom-up':
                laterals_ = [0 for i in range(len(laterals))]
                for i in range(used_backbone_levels - 1, -1, -1):
                    h, w = laterals[i].size(2), laterals[i].size(3)
                    for j in range(i+1):
                        for k in range(i-j):
                            if k == 0:
                                tmp_lateral = F.max_pool2d(laterals[j], 2, stride=2)
                            else:
                                tmp_lateral = F.max_pool2d(tmp_lateral, 2, stride=2)
                        if i > j:
                            laterals_[i] += F.interpolate(tmp_lateral, size=(h,w), mode='bilinear', align_corners=True)
                        else:
                            laterals_[i] += F.interpolate(laterals[j], size=(h,w), mode='bilinear', align_corners=True)
                laterals = laterals_
        else:
            for i in range(used_backbone_levels - 1, 0, -1):
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        if self.fpn_kernel == 1 or self.fpn_kernel == 3:    
            outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        else:
            outs = [laterals[i] for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)