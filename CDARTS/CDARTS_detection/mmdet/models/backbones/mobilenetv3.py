import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models import resume_checkpoint
from .builder import *
from ..registry import BACKBONES

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten
        if pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                assert False, 'Invalid pool type: %s' % pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'

def create_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.
    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return m

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

def conv_bn(inp, oup, stride, groups=1, act_fn=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        act_fn(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, act_fn=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        act_fn(inplace=True)
    )



default_cfgs = {
    'mobilenetv3_large_075': _cfg(url=''),
    'mobilenetv3_large_100': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
    'mobilenetv3_small_075': _cfg(url=''),
    'mobilenetv3_small_100': _cfg(url=''),
    'mobilenetv3_rw': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        interpolation='bicubic'),
    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
}

_DEBUG = False


class ChildNet(nn.Module):

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', pool_bn=False, zero_gamma=False):
        super(ChildNet, self).__init__()

        norm_layer = nn.SyncBatchNorm 
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.pool_bn = pool_bn

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        if pool_bn:
            self.pool_bn = nn.BatchNorm1d(1)

        efficientnet_init_weights(self, zero_gamma=zero_gamma)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if self.num_classes else None

    def forward_features(self, x):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        outputs = []
        # 24, 40, 96, 320
        block_idxs = [1, 2, 4, 6]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in block_idxs:
                outputs.append(x)

        # x = self.blocks(x)
        return tuple(outputs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def modify_block_args(block_args, kernel_size, exp_ratio):
    # kernel_size: 3,5,7
    # exp_ratio: 4,6
    block_type = block_args['block_type']
        # each type of block has different valid arguments, fill accordingly
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args


def _gen_childnet(**kwargs):
    # 390M
    arch_list = [[0], [3, 4, 2, 0], [5, 2, 4, 0], [4, 3, 2, 2], [1, 3, 0, 1], [2, 4, 4, 2], [0]]
    # 290M
    # arch_list = [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]]

    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x,y] for x in choices['kernel_size'] for y in choices['exp_ratio']]

    num_features = 1280

    # act_layer = HardSwish
    act_layer = Swish



    arch_def = [
       # stage 0, 112x112 in
       ['ds_r1_k3_s1_e1_c16_se0.25'],
       # stage 1, 112x112 in
       ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25'],
       # stage 2, 56x56 in
       ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
       # stage 3, 28x28 in
       ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r2_k3_s1_e4_c80_se0.25'],
       # stage 4, 14x14in
       ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
       # stage 5, 14x14in
       ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
       # stage 6, 7x7 in
       ['cn_r1_k1_s1_c960_se0.25'],
    ]

    # arch_def = [
    #     # stage 0, 112x112 in
    #     ['ds_r1_k3_s1_e1_c16_se0.25'],
    #     # stage 1, 112x112 in
    #     ['ir_r1_k3_s2_e4_c24_se0.25'],
    #     # stage 2, 56x56 in
    #     ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
    #     # stage 3, 28x28 in
    #     ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25'],
    #     # stage 4, 14x14in
    #     ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
    #         'ir_r1_k3_s1_e6_c96_se0.25'],
    #     # stage 5, 14x14in
    #     ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
    #     # stage 6, 7x7 in
    #     ['cn_r1_k1_s1_c960_se0.25'],
    # ]
    
    new_arch = []
    # change to child arch_def
    for i, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for j, (block_choice, block_arch) in enumerate(zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split('_')
                block_arch = block_arch.replace(elements[2], 'k{}'.format(str(kernel_size)))
                block_arch = block_arch.replace(elements[4], 'e{}'.format(str(exp_ratio)))
                new_layer.append(block_arch)
            new_arch.append(new_layer)

    model_kwargs = dict(
        block_args=decode_arch_def(new_arch),
        num_features=num_features,
        stem_size=16,
        # channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        num_classes=1000,
        drop_rate=0.2,
        drop_path_rate=0.2,
        global_pool='avg'
    )
    model = ChildNet(**model_kwargs)
    return model

@BACKBONES.register_module
class SSDMobilenetV3(nn.Module):
    def __init__(self, input_size, width_mult=1.0,
                 activation_type='relu',
                 single_scale=False):
        super(SSDMobilenetV3, self).__init__()
        self.input_size = input_size
        self.single_scale = single_scale
        self.width_mult = width_mult
        self.backbone = _gen_childnet()
        # del self.backbone.blocks[3][2]
        # del self.backbone.blocks[3][4]

        #for m in self.backbone.modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        m.eval()
        #        m.weight.requires_grad = False
        #        m.bias.requires_grad = False

        self.last_channel = self.backbone.blocks[-1][-1].conv.out_channels # self.backbone.blocks[-1][-1]

        # building last several layers
        self.extra_convs = []
        if not self.single_scale:
            self.extra_convs.append(conv_1x1_bn(self.last_channel, 1280,
                                                act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(1280, 256,
                                                act_fn=Swish))
            self.extra_convs.append(conv_bn(256, 256, 2, groups=256,
                                            act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(256, 512, groups=1,
                                                act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(512, 128,
                                                act_fn=Swish))
            self.extra_convs.append(conv_bn(128, 128, 2, groups=128,
                                            act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(128, 256,
                                                act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(256, 128,
                                                act_fn=Swish))
            self.extra_convs.append(conv_bn(128, 128, 2, groups=128,
                                            act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(128, 256,
                                                act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(256, 64,
                                                act_fn=Swish))
            self.extra_convs.append(conv_bn(64, 64, 2, groups=64,
                                            act_fn=Swish))
            self.extra_convs.append(conv_1x1_bn(64, 128,
                                                act_fn=Swish))
            self.extra_convs = nn.Sequential(*self.extra_convs)

    def init_weights(self, pretrained=None):
        if pretrained:
            state_dict = torch.load(pretrained)
            state_dict = state_dict['state_dict']
            # resume_checkpoint(self.backbone, pretrained)
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            print("No pretrained model!")
            return

    def forward(self, x):
        outputs = self.backbone(x)
        x = outputs[-1]
        outs = []
        for i, conv in enumerate(self.extra_convs):
            x = conv(x)
            if i % 3 == 0:
                outs.append(x)

        if self.single_scale:
            # outs.append(x)
            return outputs

        return tuple(outs)


