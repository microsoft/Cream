import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, constant_init, xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule

@NECKS.register_module
class PAFPN(nn.Module):
    r""" PAFPN Arch
        lateral      TD    3x3    BU
    C5 --------> C5     P5     N5    N5
        lateral    
    C4 --------> C4     P4     N4    N4
        lateral    
    C3 --------> C3     P3     N3    N3
        lateral    
    C2 --------> C2     P2     N2    N2
    """
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
                 lateral_kernel=1,
                 fpn_kernel=3,
                 bottom_up_kernel=3,
                 pa_kernel=3):
        super(PAFPN, self).__init__()
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
        self.bottom_up_kernel = bottom_up_kernel
        self.pa_kernel = pa_kernel

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
        self.bottom_up_convs = nn.ModuleList()
        self.pa_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):  # Faster [0,4]
            l_conv = ConvModule(
                in_channels[i], out_channels, lateral_kernel,
                padding=(lateral_kernel-1)//2, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                activation=None, inplace=True)
            fpn_conv = ConvModule(
                out_channels, out_channels, fpn_kernel,
                padding=(fpn_kernel-1)//2, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                activation=None, inplace=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        for i in range(self.start_level, self.backbone_end_level - 1):  # Faster [0,3]
            if bottom_up_kernel > 0:
                bottom_up_conv = ConvModule(
                    out_channels, out_channels, bottom_up_kernel, stride=2,
                    padding=(bottom_up_kernel-1)//2, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                    activation=activation, inplace=True)
                
                self.bottom_up_convs.append(bottom_up_conv)

            if pa_kernel > 0:    
                pa_conv = ConvModule(
                    out_channels, out_channels, pa_kernel,
                    padding=(pa_kernel-1)//2, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                    activation=activation, inplace=True)
                
                self.pa_convs.append(pa_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels, out_channels, 3,
                    stride=2, padding=1, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, activation=self.activation, inplace=True)                
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        # inputs [C2, C3, C4, C5]
        assert len(inputs) == len(self.in_channels)

        # build top-down laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)  # Faster rcnn:4

        # Top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            
        fpn_middle = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]

        # Bottom-up path
        # build outputs
        if self.pa_kernel > 0:
            outs = [fpn_middle[0]]
            for i in range(0, self.backbone_end_level - self.start_level - 1):  # Faster: [0,3]
                if self.bottom_up_kernel > 0:
                    tmp = self.bottom_up_convs[i](outs[i]) + fpn_middle[i + 1]
                else:
                    tmp = F.max_pool2d(outs[i], 2, stride=2) + fpn_middle[i + 1]
                outs.append(self.pa_convs[i](tmp))
        else:
            outs = fpn_middle

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