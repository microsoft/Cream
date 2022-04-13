import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs,\
    Mlp, PatchEmbed
try:
    from timm.models.vision_transformer import HybridEmbed
except ImportError:
    # for higher version of timm
    from timm.models.vision_transformer_hybrid import HybridEmbed

from irpe import build_rpe


class RepeatedModuleList(nn.Module):
    def __init__(self, instances, repeated_times):
        super().__init__()
        assert len(instances) == repeated_times
        self.instances = nn.ModuleList(instances)
        self.repeated_times = repeated_times
    def forward(self, *args, **kwargs):
        r = self._repeated_id
        return self.instances[r](*args, **kwargs)
    def __repr__(self):
        msg = super().__repr__()
        msg += f'(repeated_times={self.repeated_times})'
        return msg


class MiniAttention(nn.Module):
    '''
    Attention with image relative position encoding
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe_config=None, repeated_times=1, use_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        rpe_qkvs = []
        for _ in range(repeated_times):
            rpe_qkv = build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads)
            rpe_qkvs.append(rpe_qkv)
        assert len(rpe_qkvs) == repeated_times
        assert all(len(r) == 3 for r in rpe_qkvs)
        rpe_q, rpe_k, rpe_v = zip(*rpe_qkvs)
        if rpe_q[0] is not None:
            self.rpe_q = RepeatedModuleList(rpe_q, repeated_times)
        else:
            self.rpe_q = None
        if rpe_k[0] is not None:
            self.rpe_k = RepeatedModuleList(rpe_k, repeated_times)
        else:
            self.rpe_k = None
        if rpe_v[0] is not None:
            self.rpe_v = RepeatedModuleList(rpe_v, repeated_times)
        else:
            self.rpe_v = None

        if use_transform:
            transform_bias = False
            self.conv_l = RepeatedModuleList([nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=transform_bias) \
                    for _ in range(repeated_times)], repeated_times)
            self.conv_w = RepeatedModuleList([nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=transform_bias) \
                    for _ in range(repeated_times)], repeated_times)
        else:
            self.conv_l = self.conv_w = None


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        if self.conv_l is not None:
            attn = self.conv_l(attn)

        attn = attn.softmax(dim=-1)

        if self.conv_w is not None:
            attn = self.conv_w(attn)

        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)


        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in [self.conv_l, self.conv_w]:
            if m is not None:
                m.apply(_init_weights)


class MiniBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_paths=[0.], act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpe_config=None, repeated_times=1, use_transform=False):
        super().__init__()
        assert len(drop_paths) == repeated_times

        if repeated_times > 1:
            self.norm1 = RepeatedModuleList([norm_layer(dim) for _ in range(repeated_times)], repeated_times)
            self.norm2 = RepeatedModuleList([norm_layer(dim) for _ in range(repeated_times)], repeated_times)

        self.attn = MiniAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe_config=rpe_config,
            repeated_times=repeated_times,
            use_transform=use_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_paths = nn.ModuleList([DropPath(drop_path) if drop_path > 0. else nn.Identity() for drop_path in drop_paths])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        drop_path = self.drop_paths[self._repeated_id]
        x = x + drop_path(self.attn(self.norm1(x)))
        x = x + drop_path(self.mlp(self.norm2(x)))
        return x


class RepeatedMiniBlock(nn.Module):
    def __init__(self, repeated_times, **kwargs):
        super().__init__()
        self.repeated_times = repeated_times
        self.block = MiniBlock(repeated_times=repeated_times, **kwargs)

        def set_repeated_times_fn(m):
            m._repeated_times = repeated_times
        self.apply(set_repeated_times_fn)

    def forward(self, x):
        for i, t in enumerate(range(self.repeated_times)):
            def set_repeated_id(m):
                m._repeated_id = i
            self.block.apply(set_repeated_id)
            x = self.block(x)
        return x

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(repeated_times={self.repeated_times})'
        return msg


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
                           and image relative position encoding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, rpe_config=None,
                 use_cls_token=True,
                 repeated_times=1,
                 use_transform=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        pos_embed_len = 1 + num_patches if use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        assert depth % repeated_times == 0
        depth //= repeated_times

        blocks = []

        block_kwargs = dict(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, rpe_config=rpe_config,
                use_transform=use_transform)

        for i in range(depth):
            if repeated_times > 1:
                block = RepeatedMiniBlock(
                        repeated_times=repeated_times,
                        drop_paths=dpr[i * repeated_times : (i + 1) * repeated_times],
                        **block_kwargs,
                    )
            else:
                block = MiniBlock(drop_paths=[dpr[i]], **block_kwargs)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if not use_cls_token:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.avgpool = None

        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.apply(self._init_custom_weights)

        def set_repeated_id(m):
            m._repeated_id = 0
        self.apply(set_repeated_id)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_custom_weights(self, m):
        if hasattr(m, 'init_weights'):
            m.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.cls_token is not None:
            return x[:, 0]
        else:
            return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.avgpool is not None:
            x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
            x = torch.flatten(x, 1)
        x = self.head(x)
        return x
