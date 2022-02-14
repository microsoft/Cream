import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
# from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from mmdet.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 kernel_size=3,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv2_split=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 kernel_size=3,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv2_split=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv2_split = conv2_split
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            if not self.conv2_split:
                self.conv2 = build_conv_layer(
                    conv_cfg,
                    planes,
                    planes,
                    kernel_size=kernel_size,
                    stride=self.conv2_stride,
                    padding=int((kernel_size-1)*dilation/2),
                    dilation=dilation,
                    bias=False)
            else:
                self.conv2_d1 = build_conv_layer(
                    conv_cfg, planes, planes-2*int(planes/3), kernel_size=3,
                    stride=self.conv2_stride, padding=dilation,
                    dilation=dilation, bias=False)
                self.conv2_d2 = build_conv_layer(
                    conv_cfg, planes, int(planes/3), kernel_size=3,
                    stride=self.conv2_stride, padding=dilation+1,
                    dilation=dilation+1, bias=False)
                self.conv2_d3 = build_conv_layer(
                    conv_cfg, planes, int(planes/3), kernel_size=3,
                    stride=self.conv2_stride, padding=dilation+2,
                    dilation=dilation+2, bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            """ original code
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            """
            # for huang lang test
            self.conv2_offset = StructualDeformBlock(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                if not self.conv2_split:
                    out = self.conv2(out)
                else:
                    out_d1 = self.conv2_d1(out)
                    out_d2 = self.conv2_d2(out)
                    out_d3 = self.conv2_d3(out)
                    out = torch.cat((out_d1, out_d2, out_d3), 1)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# for huang lang test
from torch.nn import functional as F
class StructualDeformBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(StructualDeformBlock, self).__init__()
        assert out_channels == 2 * kernel_size**2
        self.out_channels = out_channels + 6
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # conv weights
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))

        # homogeneous coordinate map
        coord = (torch.arange(kernel_size, dtype=torch.float) - kernel_size // 2) * dilation
        coord = list(torch.meshgrid([coord, coord]))
        coord.append(torch.ones(kernel_size, kernel_size))
        self.coord_map = torch.autograd.Variable(torch.stack(coord, dim=0).view(3, -1), requires_grad=False) # (3, K**2)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, padding={padding}, dilation={dilation}')
        return s.format(**self.__dict__)
    
    def forward(self, x_):
        offset_affine = F.conv2d(x_, self.weight, self.bias, self.stride, self.padding, self.dilation)
        n, c, h, w = offset_affine.shape
        # apply affine transformation on conv grids
        deform_params = offset_affine[:, -6:].view(n, 2, 3, h, w)
        structural_offset = torch.einsum('nijhw,jk->nikhw', (deform_params, self.coord_map.to(deform_params.device)))
        offset = structural_offset.reshape(n, -1, h, w) + offset_affine[:, :-6]
        return offset


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
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, dilation=1, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels)
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out = out + x
        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   kernel_size=3,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv2_split=False,
                   toy_replace=None,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            kernel_size=kernel_size,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv2_split=conv2_split,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        if blocks > 30 and i % 2 == 1:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    kernel_size=3,
                    dilation=2,
                    style=style,
                    with_cp=with_cp,
                    conv2_split=conv2_split,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    gcb=gcb,
                    gen_attention=gen_attention if
                    (i in gen_attention_blocks) else None))
        elif toy_replace is not None and i == toy_replace.get('layer', 30):
            if toy_replace.get('block', 'res') == 'ir':
                layers.append(
                    MBBlock(inplanes, inplanes, 1, 1, toy_replace.get('conv_kernel'), dilation=toy_replace.get('dilation'), groups=1)
                )
            else: 
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        kernel_size=toy_replace.get('conv_kernel'),
                        dilation=toy_replace.get('dilation'),
                        style=style,
                        with_cp=with_cp,
                        conv2_split=conv2_split,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        dcn=dcn,
                        gcb=gcb,
                        gen_attention=gen_attention if
                        (i in gen_attention_blocks) else None))
        else:   
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    style=style,
                    with_cp=with_cp,
                    conv2_split=conv2_split,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    gcb=gcb,
                    gen_attention=gen_attention if
                    (i in gen_attention_blocks) else None))
    # for [1,2,3,1]
    '''
    layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                kernel_size=3,
                dilation=2,
                style=style,
                with_cp=with_cp,
                conv2_split=conv2_split,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))
    '''
    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 kernel_size=3,
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 conv2_split=False,
                 toy_replace=None, # for toy experiments replace
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.kernel_size=kernel_size
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.conv2_split = conv2_split
        self.toy_replace = toy_replace
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        _toy_replace = None
        for i, num_blocks in enumerate(self.stage_blocks):
            if self.toy_replace is not None:
                if i != self.toy_replace.get('stage'):
                    _toy_replace = None
                else:
                    _toy_replace = self.toy_replace
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                kernel_size=kernel_size,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv2_split=conv2_split,
                toy_replace=_toy_replace,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

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

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


from collections import OrderedDict

def load_checkpoint(model,
                    filename,
                    strict=False,
                    logger=None):


    checkpoint = torch.load(filename)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        omit_name = None
        if model.toy_replace is not None:
            # layer3.1.conv2.weight
            omit_name = 'layer' + str(model.toy_replace.get('stage')+1) + '.' + str(model.toy_replace.get('layer')) + '.conv2.weight'
        load_state_dict(model, state_dict, strict, logger, omit_name=omit_name)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None, omit_name=None):
    """Load state_dict to a module.
    Args:
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    state_dict_modify = state_dict.copy()
    for name, param in state_dict.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if 'conv2' in name and 'layer4.0.conv2_d2.weight' in own_state.keys():
            d1 = name.replace('conv2', 'conv2_d1')
            d1_c = own_state[d1].size(0)
            own_state[d1].copy_(param[:d1_c,:,:,:])
            state_dict_modify[d1] = param[:d1_c,:,:,:]

            d2 = name.replace('conv2', 'conv2_d2')
            d2_c = own_state[d2].size(0)
            own_state[d2].copy_(param[d1_c:d1_c+d2_c,:,:,:])
            state_dict_modify[d2] = param[d1_c:d1_c+d2_c,:,:,:]

            d3 = name.replace('conv2', 'conv2_d3')
            own_state[d3].copy_(param[d1_c+d2_c:,:,:,:])
            state_dict_modify[d3] = param[d1_c+d2_c:,:,:,:]
        else:
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            try:
                if name == omit_name:
                    print('{} is omitted.'.format(omit_name))
                else:
                    own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)