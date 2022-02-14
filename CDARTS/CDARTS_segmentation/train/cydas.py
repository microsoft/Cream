import torch
import torch.nn as nn
from torch.nn import functional as F
from builder import *
from operations import *
from operations import DropPath_
from genotypes import PRIMITIVES
from pdb import set_trace as bp
from seg_oprs import FeatureFusion, Head, Decoder
from layers import NaiveSyncBatchNorm

# BatchNorm2d = nn.BatchNorm2d
BatchNorm2d = NaiveSyncBatchNorm

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

        norm_layer = BatchNorm2d
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
        # 16, 24, 40, 96, 320
        # block_idxs = [0, 1, 2, 4, 6]
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
    # arch_list = [[0], [3, 2], [3, 2], [3, 3], [3, 3, 3], [3, 3, 3], [0]]
    # arch_list = [[0], [3, 2, 3, 3], [3, 2, 3, 1], [3, 0, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [0]]
    # arch_list = [[0], [3,4,3,1],[3,2,3,0],[3,3,3,1],[3,3,3,3],[3,3,3,3],[0]]
    arch_list = [[0], [3, 4, 2, 0], [5, 2, 4, 0], [4, 3, 2, 2], [1, 3, 0, 1], [2, 4, 4, 2], [0]]
    # arch_list = [[0], [], [], [], [], [0]]
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x,y] for x in choices['kernel_size'] for y in choices['exp_ratio']]

    num_features = 1280

    # act_layer = HardSwish
    act_layer = Swish
    '''
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r2_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]
    '''
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
        ['cn_r1_k1_s1_c320_se0.25'],
    ]
    #arch_def = [ 
    #    # stage 0, 112x112 in 
    #    ['ds_r1_k3_s1_e1_c16_se0.25'], 
    #    # stage 1, 112x112 in 
    #    ['ir_r1_k3_s2_e4_c24_se0.25'], 
    #    # stage 2, 56x56 in 
    #    ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'], 
    #    # stage 3, 28x28 in 
    #    ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s2_e6_c80_se0.25'], 
    #    # stage 4, 14x14in 
    #    ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'], 
    #    # stage 5, 14x14in 
    #    ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'], 
    #    # stage 6, 7x7 in 
    #    ['cn_r1_k1_s1_c320_se0.25'], 
    #] 


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

class CyDASseg(nn.Module):
    def __init__(self, Fch=12, num_classes=19, stem_head_width=(1., 1.)):
        super(CyDASseg, self).__init__()
        self._num_classes = num_classes
        self._stem_head_width = stem_head_width
        self.backbone = _gen_childnet()
        # self.f_channels = [16, 24, 40, 96]
        self.f_channels = [24, 40, 96, 320]
        self._Fch = Fch
        # del self.backbone.blocks[3][2]

        #for m in self.backbone.modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        m.eval()
        #        m.weight.requires_grad = False
        #        m.bias.requires_grad = False

        self.last_channel = self.backbone.blocks[-1][-1].conv.out_channels # self.backbone.blocks[-1][-1]

        # building decoder
        self.build_arm_ffm_head()

    def init_weights(self, pretrained=None):
        if pretrained:
            state_dict = torch.load(pretrained)
            state_dict = state_dict['state_dict']
            # resume_checkpoint(self.backbone, pretrained)
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            print("No pretrained model!")
            return

    def build_arm_ffm_head(self):

        # 24, 40, 96, 320

        if self.training:
            self.heads32 = Head(self.f_channels[-1], self._num_classes, True, norm_layer=BatchNorm2d)
            self.heads16 = Head(self.f_channels[-2], self._num_classes, True, norm_layer=BatchNorm2d)

        self.heads8 = Decoder(self.num_filters(8, self._stem_head_width[1]), self.f_channels[0], self._num_classes, Fch=self._Fch, scale=4, branch=1, is_aux=False, norm_layer=BatchNorm2d)

        self.arms32 = nn.ModuleList([
            ConvNorm(self.f_channels[-1], self.num_filters(16, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
            ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
        ])

        self.refines32 = nn.ModuleList([
            ConvNorm(self.num_filters(16, self._stem_head_width[1])+self.f_channels[-2], self.num_filters(16, self._stem_head_width[1]), 3, 1, 1, slimmable=False),
            ConvNorm(self.num_filters(8, self._stem_head_width[1])+self.f_channels[-3], self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, slimmable=False),
        ])


        self.ffm = FeatureFusion(self.num_filters(8, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=8, branch=1, norm_layer=BatchNorm2d)

    def agg_ffm(self, outputs8, outputs16, outputs32, outputs4):
        pred32 = []; pred16 = []; pred8 = [] # order of predictions is not important

        if self.training: pred32.append(outputs32)
        out = self.arms32[0](outputs32)
        out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=False)
        out = self.refines32[0](torch.cat([out, outputs16], dim=1))
        if self.training: pred16.append(outputs16)
        out = self.arms32[1](out)
        out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=False)
        out = self.refines32[1](torch.cat([out, outputs8], dim=1))
        pred8.append(out)

        if len(pred32) > 0:
            pred32 = self.heads32(torch.cat(pred32, dim=1))
        else:
            pred32 = None
        if len(pred16) > 0:
            pred16 = self.heads16(torch.cat(pred16, dim=1))
        else:
            pred16 = None
        pred8 = self.heads8(self.ffm(torch.cat(pred8, dim=1)), outputs4)
        if self.training: 
            return pred8, pred16, pred32
        else:
            return pred8

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def forward(self, x):
        b,c,h,w = x.shape
        outputs = self.backbone(x)
        
        outputs4, outputs8, outputs16, outputs32 =outputs[0], outputs[1], outputs[2], outputs[3]
        if self.training:
            pred8, pred16, pred32 = self.agg_ffm(outputs8, outputs16, outputs32, outputs4)
            pred8 = F.interpolate(pred8, size=(h,w), mode='bilinear', align_corners=False)
            if pred16 is not None: pred16 = F.interpolate(pred16, size=(h,w), mode='bilinear', align_corners=False)
            if pred32 is not None: pred32 = F.interpolate(pred32, size=(h,w), mode='bilinear', align_corners=False)
            return pred8, pred16, pred32
        else:
            pred8 = self.agg_ffm(outputs8, outputs16, outputs32, outputs4)
            out = F.interpolate(pred8, size=(int(pred8.size(2))*4, int(pred8.size(3))*4), mode='bilinear', align_corners=False)
            return out



