""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import functools
import inspect
from copy import deepcopy
import os
import random
import copy
from contextlib import nullcontext
from argparse import Namespace

from dataclasses import dataclass
import functools
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
# apply the non-reentrant variant of checkpoint
if 'use_reentrant' in inspect.signature(checkpoint).parameters:
    checkpoint = functools.partial(checkpoint, use_reentrant=False)

from .timm_model import TimmModel
from .utils import freeze_batch_norm_2d, to_2tuple
from .resnet import ModifiedResNet
from .l0module import L0Module


def load_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=True)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, hidden_z=None):
        '''
        x: (N, L, C)
        hidden_z: (C,)
        '''
        self.hidden_z = hidden_z
        orig_type = x.dtype

        if hidden_z is None:
            x = F.layer_norm(x, self.normalized_shape,
                             self.weight, self.bias, self.eps)
        else:
            assert len(self.normalized_shape) == 1
            # [TODO] weighted layer norm
            remaining_index = torch.where(hidden_z != 0)[0]
            compressed_input = torch.index_select(
                x, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            x = x.new_zeros(x.shape)
            x[..., remaining_index] = normed_input.to(orig_type)

        return x.to(orig_type)

    def prune(self):
        if self.hidden_z is None:
            return self
        hidden_z = self.hidden_z
        assert len(self.normalized_shape) == 1
        remaining_index = torch.where(hidden_z != 0)[0]
        compressed_weight = self.weight[remaining_index]
        compressed_bias = self.bias[remaining_index]
        # m = self
        m = LayerNorm(remaining_index.shape[0]).to(self.weight.device)
        m.normalized_shape = (len(remaining_index),)
        m.weight.data = compressed_weight.contiguous()
        m.bias.data = compressed_bias.contiguous()
        return m

    def prune_mul_hidden(self):
        if self.hidden_z is None:
            return self
        hidden_z = self.hidden_z
        assert len(self.normalized_shape) == 1
        remaining_index = torch.where(hidden_z != 0)[0]
        compressed_weight = self.weight[remaining_index] * \
            hidden_z[remaining_index]
        compressed_bias = self.bias[remaining_index] * \
            hidden_z[remaining_index]
        m = self
        m.normalized_shape = (len(remaining_index),)
        m.weight.data = compressed_weight.contiguous()
        m.bias.data = compressed_bias.contiguous()
        return m


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, d_model, mlp_width, act_layer=nn.GELU, scale_fc=False):
        super().__init__()
        self.d_model = d_model
        self.mlp_width = mlp_width
        self.c_fc = nn.Linear(d_model, mlp_width)
        assert not scale_fc
        # self.ln = LayerNorm(mlp_width) if scale_fc else nn.Identity()
        self.act_layer = act_layer
        self.scale_fc = scale_fc
        self.gelu = act_layer()
        self.c_proj = nn.Linear(mlp_width, d_model)

    def forward(self, x, hidden_z=None, intermediate_z=None):
        '''
        x: (N, L, C)
        intermediate_z: (mlp_width,) or (1, 1, mlp_width)
        hidden_z: (embed_dim,) or (1, 1, embed_dim)
        '''
        self.hidden_z = hidden_z
        self.intermediate_z = intermediate_z

        x = self.c_fc(x)
        x = self.gelu(x)
        if intermediate_z is not None:
            x = torch.mul(x, intermediate_z)
        x = self.c_proj(x)
        if hidden_z is not None:
            x = torch.mul(x, hidden_z)
        return x

    def prune(self):
        device = self.c_fc.weight.device
        if self.hidden_z is None:
            self.hidden_z = torch.ones(
                (self.d_model,), dtype=torch.bool, device=device)
        if self.intermediate_z is None:
            self.intermediate_z = torch.ones(
                (self.mlp_width,), dtype=torch.bool, device=device)
        hidden_r = torch.where(self.hidden_z != 0)[0]
        intermediate_r = torch.where(self.intermediate_z != 0)[0]
        d_model = len(hidden_r)
        mlp_width = len(intermediate_r)
        # m = self
        m = copy.deepcopy(self)
        m.c_fc = nn.Linear(hidden_r.shape[0], intermediate_r.shape[0])
        m.c_proj = nn.Linear(intermediate_r.shape[0], hidden_r.shape[0])
        m.d_model = d_model
        m.mlp_width = mlp_width
        m.c_fc.weight = nn.Parameter(
            (self.c_fc.weight[intermediate_r][:, hidden_r]).contiguous())
        m.c_fc.bias = nn.Parameter(
            (self.c_fc.bias[intermediate_r]).contiguous())

        m.c_proj.weight = nn.Parameter(((self.c_proj.weight *
                                         self.intermediate_z.view(1, -1) * self.hidden_z.view(-1, 1))[hidden_r][:, intermediate_r]).contiguous())
        m.c_proj.bias = nn.Parameter(
            ((self.c_proj.bias * self.hidden_z)[hidden_r]).contiguous())
        return m


class MultiheadAttention(nn.MultiheadAttention):
    def prune(self):
        device = self.in_proj_weight.device
        if self.hidden_z is None:
            self.hidden_z = torch.ones(
                (self.embed_dim,), dtype=torch.bool, device=device)
        if self.head_z is None:
            self.head_z = torch.ones(
                (self.num_heads,), dtype=torch.bool, device=device)
        hidden_r = torch.where(self.hidden_z != 0)[0]
        head_r = torch.where(self.head_z != 0)[0]
        d_model = len(hidden_r)
        d_head = len(head_r)
        org_num_heads = self.num_heads
        org_head_dim = self.head_dim
        org_embed_dim = self.embed_dim
        mod = self
        mod.use_naive_compute = True
        mod.embed_dim = d_model
        mod.head_dim = self.head_dim
        mod.num_heads = d_head
        inter_dim = d_head * self.head_dim
        mod.in_proj_weight = nn.Parameter(self.in_proj_weight.view(
            3, org_num_heads, org_head_dim, org_embed_dim)[:, head_r][..., hidden_r].reshape(-1, d_model))
        if self.in_proj_bias is not None:
            mod.in_proj_bias = nn.Parameter(self.in_proj_bias.view(
                3, org_num_heads, org_head_dim)[:, head_r].reshape(-1))
        mod.out_proj.weight = nn.Parameter(
            ((self.out_proj.weight * self.hidden_z.view(-1, 1)).
             view(org_embed_dim, org_num_heads, org_head_dim) * self.head_z.view(1, org_num_heads, 1))[hidden_r][:, head_r].reshape(d_model, -1)
        )
        if self.out_proj.bias is not None:
            mod.out_proj.bias = nn.Parameter(
                (self.out_proj.bias * self.hidden_z.view(-1,)).
                view(org_embed_dim)[hidden_r].reshape(-1)
            )
        return mod


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = MultiheadAttention(d_model, n_head)
        assert not scale_attn
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = Mlp(d_model, mlp_width, act_layer, scale_fc)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                  *,
                  head_z: Optional[torch.Tensor] = None,
                  hidden_z: Optional[torch.Tensor] = None,
                  ):

        self.attn.head_z = head_z
        self.attn.hidden_z = hidden_z

        if (head_z is None and hidden_z is None and
                not getattr(self.attn, 'use_naive_compute', False)):
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        else:
            # the following code does not support `attn_mask`
            # x: (length, batch_size, embed_dim)
            n_head = self.attn.num_heads
            length, batch_size, d_model = x.shape
            ws = self.attn.in_proj_weight.chunk(3)
            bs = self.attn.in_proj_bias.chunk(3)
            dim_per_head = len(ws[0]) // n_head
            # (length, batch_size, n_head * dim_per_head)
            q, k, v = [F.linear(x, w, b) for w, b in zip(ws, bs)]
            # (batch_size * n_head, length, d_head)
            q = q.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            k = k.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            v = v.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            scale = dim_per_head ** -0.5
            q *= scale
            # (batch_size * n_head, length, length)
            sim = q @ k.transpose(1, 2)
            if attn_mask is not None:
                sim += attn_mask
            sim = torch.softmax(sim, -1)
            # (batch_size * n_head, length, head_dim)
            out = sim @ v
            if head_z is not None:
                out = out.view(batch_size, n_head, length, dim_per_head)
                # head_z: (1, n_head, 1, 1)
                out *= head_z.view(1, -1, 1, 1)
                out = out.view(batch_size * n_head, length, dim_per_head)
            out = out.transpose(0, 1).reshape(length, batch_size, -1)
            out = F.linear(out, self.attn.out_proj.weight,
                           self.attn.out_proj.bias)
            if hidden_z is not None:
                out = torch.mul(out, hidden_z)
            return out

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None):

        self.hidden_z = hidden_z
        self.heads_z = heads_z
        self.mha_z = mha_z
        self.intermediate_z = intermediate_z
        self.ffn_z = ffn_z

        # x: (length, batch_size, embed_dim) e.g. 50, 128, 768 for vision
        if self.attention is not None:
            attn_out = self.attention(self.ln_1(x, hidden_z=hidden_z),
                                      attn_mask=attn_mask,
                                      head_z=heads_z, hidden_z=hidden_z)
            if mha_z is not None:  # a number
                attn_out = attn_out.mul(mha_z)
            x = x + attn_out
        if self.mlp is not None:
            ln_2_out = self.ln_2(x, hidden_z=hidden_z)

            mlp_out = self.mlp(ln_2_out,
                               intermediate_z=intermediate_z,
                               hidden_z=hidden_z)
            if ffn_z is not None:  # a number
                mlp_out = mlp_out.mul(ffn_z)
            x = x + mlp_out
        return x

    def prune(self):
        mod = self
        if (self.mha_z is not None and self.mha_z.item() == 0) or (self.heads_z).sum() == 0:
            mod.ln_1 = None
            mod.attn = None
            mod.attention = None
        else:
            mod.ln_1 = mod.ln_1.prune()
            mod.attn = mod.attn.prune()
            if self.mha_z is not None:
                mod.attn.out_proj.weight.data *= self.mha_z
                mod.attn.out_proj.bias.data *= self.mha_z

        if self.ffn_z is not None and self.ffn_z.item() == 0:
            mod.ln_2 = None
            mod.mlp = None
        else:
            mod.ln_2 = mod.ln_2.prune()
            mod.mlp = mod.mlp.prune()
            if self.ffn_z is not None:
                mod.mlp.c_proj.weight.data *= self.ffn_z
                mod.mlp.c_proj.bias.data *= self.ffn_z
        return mod


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        assert width % heads == 0
        self.head_dim = width // heads
        self.num_heads = heads
        self.mlp_ratio = mlp_ratio

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None):

        return self.infer_blocks(x, attn_mask,
                                 hidden_z=hidden_z,
                                 heads_z=heads_z,
                                 mha_z=mha_z,
                                 intermediate_z=intermediate_z,
                                 ffn_z=ffn_z)

    def infer_blocks(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, block_idxs=None,
                     hidden_z: Optional[torch.Tensor] = None,
                     heads_z: Optional[torch.Tensor] = None,
                     mha_z: Optional[torch.Tensor] = None,
                     intermediate_z: Optional[torch.Tensor] = None,
                     ffn_z: Optional[torch.Tensor] = None):

        num_layers = self.layers
        if hidden_z is not None:
            assert hidden_z.shape == (self.width,)
        if heads_z is not None:
            if heads_z.ndim == 5:
                heads_z = heads_z.view(num_layers, self.num_heads)
            assert heads_z.shape in [(num_layers, self.num_heads), (self.num_heads,)], (
                heads_z.shape, (num_layers, self.num_heads))
        if mha_z is not None:
            assert mha_z.shape == (num_layers,), mha_z.shape
        if intermediate_z is not None:
            if intermediate_z.ndim == 4:
                intermediate_z = intermediate_z.view(num_layers, -1)
            assert intermediate_z.shape in [
                (num_layers, self.mlp_ratio * self.width), (self.mlp_ratio * self.width,)], intermediate_z.shape
        if ffn_z is not None:
            assert ffn_z.shape == (num_layers,), ffn_z.shape

        def _get_zi(z, i, ndim=2):
            if z is None:
                return None
            if z.ndim == ndim:
                return z[i]
            return z

        block_idxs = block_idxs or list(range(self.layers))
        for i in block_idxs:
            r = self.resblocks[i]

            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask,
                               hidden_z,
                               _get_zi(heads_z, i),
                               _get_zi(mha_z, i, ndim=1),
                               _get_zi(intermediate_z, i),
                               _get_zi(ffn_z, i, ndim=1))
            else:
                x = r(x, attn_mask=attn_mask,
                      hidden_z=hidden_z,
                      heads_z=_get_zi(heads_z, i),
                      mha_z=_get_zi(mha_z, i, ndim=1),
                      intermediate_z=_get_zi(intermediate_z, i),
                      ffn_z=_get_zi(ffn_z, i, ndim=1))

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def extra_repr(self):
        return f'grad_checkpointing={self.grad_checkpointing}'

    def prune(self):
        mod = self
        for i in range(len(self.resblocks)):
            self.resblocks[i] = self.resblocks[i].prune()
        return mod


class VisualTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU,
            teacher_width: int = -1,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.embed_dim = width
        self.layers = layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer)
        self.head_dim = width // heads

        self.ln_post = LayerNorm(width)
        # image proj
        if teacher_width > 0:
            self.proj = nn.Parameter(torch.empty(
                teacher_width, output_dim), requires_grad=False)
        else:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None,
                embed_dim_z: Optional[torch.Tensor] = None):

        self.hidden_z = hidden_z
        self.embed_dim_z = embed_dim_z

        x = x.to(self.conv1.weight.device)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # the first token is the class token.
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, 1 + grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)  # 128, 50, 768

        if hidden_z is not None:
            x = torch.mul(x, hidden_z)
        x = self.ln_pre(x, hidden_z=hidden_z)

        x = x.permute(1, 0, 2)  # NLD -> LND 50, 128, 768
        x = self.transformer(x,
                             hidden_z=hidden_z,
                             heads_z=heads_z,
                             mha_z=mha_z,
                             intermediate_z=intermediate_z,
                             ffn_z=ffn_z)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # select class token
        x = self.ln_post(x[:, 0, :], hidden_z=hidden_z)

        if self.proj is not None:
            x = self.get_proj_feature(x)

        return x

    def get_proj_feature(self, x):
        if self.proj is not None:
            x = x @ self.proj
        return x

    def extra_repr(self):
        return 'image_size={}, output_dim={}'.format(self.image_size, self.output_dim)

    def prune(self):
        hidden_r = torch.where(self.hidden_z != 0)[0]
        self.conv1.weight = nn.Parameter(
            (self.conv1.weight.data * self.hidden_z.view(-1, 1, 1, 1))[hidden_r])
        if self.conv1.bias is not None:
            self.conv1.bias = nn.Parameter(
                (self.conv1.bias * self.hidden_z.view(-1,))[hidden_r])
        self.class_embedding = nn.Parameter(
            (self.class_embedding * self.hidden_z.view(-1,))[hidden_r])
        self.positional_embedding = nn.Parameter(
            (self.positional_embedding * self.hidden_z.view(1, -1))[:, hidden_r])
        self.ln_pre = self.ln_pre.prune()
        self.transformer = self.transformer.prune()
        self.ln_post = self.ln_post.prune()
        if self.embed_dim_z is not None:
            embed_dim_r = self.embed_dim_z > 0
            self.proj = nn.Parameter((self.proj * self.hidden_z.view(-1, 1)
                                     * self.embed_dim_z.view(1, -1))[hidden_r][:, embed_dim_r])
        else:
            self.proj = nn.Parameter(
                (self.proj * self.hidden_z.view(-1, 1))[hidden_r])
        return self


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    teacher_width: int = -1
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    # use (imagenet) pretrained weights for named model
    timm_model_pretrained: bool = False
    # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_pool: str = 'avg'
    # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj: str = 'linear'


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    teacher_width: int = -1
    heads: int = 8
    layers: int = 12


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, vision_cfg, quick_gelu,
                 l0_module_image=False,
                 mask_cfg=None):
        super().__init__()
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,
                teacher_width=vision_cfg.teacher_width,
            )
        self.init_parameters()

        if l0_module_image:
            logging.info('use l0_module_vision')
            config_mask = Namespace()
            config_mask.hidden_size = vision_cfg.width
            config_mask.intermediate_size = 4 * vision_cfg.width
            config_mask.num_attention_heads = vision_heads
            config_mask.num_hidden_layers = vision_cfg.layers
            config_mask.sparsity_warmup = mask_cfg.sparsity_warmup
            config_mask.sparsity = mask_cfg.sparsity
            config_mask.start_sparsity = mask_cfg.start_sparsity
            self.l0_module = L0Module(config_mask, lagrangian_warmup=config_mask.sparsity_warmup, start_sparsity=config_mask.start_sparsity,
                                      target_sparsity=config_mask.sparsity, pruning_type=["hidden", "heads", "intermediate"])
        else:
            self.l0_module = None

        self.mask = None

    def init_parameters(self):
        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

    def forward(self, image, normalized=False,
                **mask):

        if self.l0_module is not None:
            mask = self.l0_module.forward()

        self.mask = mask

        image_features = self.visual(image, **mask)

        embed_dim_z = mask.get('embed_dim_z', None)
        if embed_dim_z is not None:
            image_features = image_features.mul(embed_dim_z)

        if normalized:
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def prune(self):
        self.visual = self.visual.prune()
        return self


class TextEncoder(nn.Module):
    def __init__(self, embed_dim, text_cfg, quick_gelu,
                 l0_module_text, mask_cfg=None):
        super().__init__()

        act_layer = QuickGELU if quick_gelu else nn.GELU
        self.context_length = text_cfg.context_length

        if text_cfg.layers > 0:
            self.transformer = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
        else:
            self.transformer = None

        self.text_projection = None
        if text_cfg.layers > 0:
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(
                text_cfg.vocab_size, text_cfg.width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, text_cfg.width))
            self.ln_final = LayerNorm(text_cfg.width)
            if text_cfg.teacher_width > 0:
                self.text_projection = nn.Parameter(torch.empty(
                    text_cfg.width, embed_dim), requires_grad=False)
            else:
                self.text_projection = nn.Parameter(
                    torch.empty(text_cfg.width, embed_dim))
            self.register_buffer(
                'attn_mask', self.build_attention_mask(), persistent=False)
        else:
            self.token_embedding = None
        self.init_parameters()

        if l0_module_text:
            logging.info('use l0_module_text')
            config_mask = Namespace()
            config_mask.hidden_size = text_cfg.width
            config_mask.intermediate_size = 4 * text_cfg.width
            config_mask.num_attention_heads = text_cfg.heads
            config_mask.num_hidden_layers = text_cfg.layers
            config_mask.sparsity_warmup = mask_cfg.sparsity_warmup
            config_mask.sparsity = mask_cfg.sparsity
            config_mask.start_sparsity = mask_cfg.start_sparsity
            self.l0_module = L0Module(config_mask, lagrangian_warmup=config_mask.sparsity_warmup, start_sparsity=config_mask.start_sparsity,
                                      target_sparsity=config_mask.sparsity, pruning_type=["hidden", "heads", "intermediate"])
        else:
            self.l0_module = None

        self.mask = None

    def init_parameters(self):
        if self.transformer is not None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)

            proj_std = (self.transformer.width ** -0.5) * \
                ((2 * self.transformer.layers) ** -0.5)
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.text_projection is not None:
                nn.init.normal_(self.text_projection,
                                std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, normalized=False,
                    hidden_z: Optional[torch.Tensor] = None,
                    heads_z: Optional[torch.Tensor] = None,
                    mha_z: Optional[torch.Tensor] = None,
                    intermediate_z: Optional[torch.Tensor] = None,
                    ffn_z: Optional[torch.Tensor] = None,
                    embed_dim_z: Optional[torch.Tensor] = None,
                    ):
        self.hidden_z = hidden_z
        self.embed_dim_z = embed_dim_z

        text = text.to(self.token_embedding.weight.device)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        if hidden_z is not None:
            x = torch.mul(x, hidden_z)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask,
                             hidden_z=hidden_z,
                             heads_z=heads_z,
                             mha_z=mha_z,
                             intermediate_z=intermediate_z,
                             ffn_z=ffn_z)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x, hidden_z)

        # if hidden_z is not None:
        #     x = torch.mul(x, hidden_z)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.get_proj_feature(x)
        if embed_dim_z is not None:
            x = x.mul(embed_dim_z)
        if normalized:
            x = F.normalize(x, dim=-1)

        return x

    def get_proj_feature(self, x):
        return x @ self.text_projection

    def forward(self, text, normalized=False):
        mask = dict()

        if self.l0_module is not None:
            mask = self.l0_module.forward()

        self.mask = mask

        return self.encode_text(text, normalized=normalized, **mask)

    def prune(self):
        device = self.token_embedding.weight.device
        if self.hidden_z is None:
            self.hidden_z = torch.ones(
                self.text_projection.size(0), device=device)
        if self.embed_dim_z is None:
            self.embed_dim_z = torch.ones(
                self.text_projection.size(1), device=device)
        mod = self
        self_copy = copy.deepcopy(self)
        hidden_r = self.hidden_z > 0
        mod.token_embedding = nn.Embedding(
            self_copy.token_embedding.weight.shape[0], hidden_r.sum())
        mod.positional_embedding = nn.Parameter(
            torch.empty(self_copy.context_length, hidden_r.sum()))
        mod.token_embedding.weight = nn.Parameter(
            (self_copy.token_embedding.weight * self_copy.hidden_z.view(1, -1))[:, hidden_r])
        mod.positional_embedding = nn.Parameter(
            (self_copy.positional_embedding * self_copy.hidden_z.view(1, -1))[:, hidden_r])
        mod.transformer = self.transformer.prune()
        mod.ln_final = self.ln_final.prune()
        embed_dim_r = self.embed_dim_z > 0
        mod.text_projection = nn.Parameter(
            (self.text_projection * self.hidden_z.view(-1, 1) * self.embed_dim_z.view(1, -1))[hidden_r][:, embed_dim_r])
        return mod


class LogitScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, dummy):
        return self.logit_scale


class FNBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class CLIPBase(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder

        self._logit_scale = LogitScale()

        # autocast context
        self.image_autocast = nullcontext
        self.text_autocast = nullcontext
        self.logit_autocast = nullcontext

        # copy the module without ddp
        self._without_ddp = [self._image_encoder,
                             self._text_encoder, self._logit_scale]

        self.used_ddp = False

    def set_autocast(self, image_autocast, text_autocast, logit_autocast):
        self.image_autocast = image_autocast
        self.text_autocast = text_autocast
        self.logit_autocast = logit_autocast

    @property
    def image_encoder_without_ddp(self):
        return self._without_ddp[0]

    @image_encoder_without_ddp.setter
    def image_encoder_without_ddp(self, encoder):
        assert self.used_ddp is False
        self._image_encoder = encoder
        self._without_ddp[0] = self._image_encoder

    @property
    def text_encoder_without_ddp(self):
        return self._without_ddp[1]

    @text_encoder_without_ddp.setter
    def text_encoder_without_ddp(self, encoder):
        assert self.used_ddp is False
        self._text_encoder = encoder
        self._without_ddp[1] = self._text_encoder

    @property
    def logit_scale_without_ddp(self):
        return self._without_ddp[2]

    @logit_scale_without_ddp.setter
    def logit_scale_without_ddp(self, logit_scale):
        assert self.used_ddp is False
        self._logit_scale = logit_scale
        self._without_ddp[2] = self._logit_scale

    @property
    def visual(self):
        return self.image_encoder_without_ddp.visual

    @property
    def transformer(self):
        return self.text_encoder_without_ddp.transformer

    @property
    def text_encoder_without_ddp(self):
        return self._without_ddp[1]

    @property
    def logit_scale_without_ddp(self):
        return self._without_ddp[2]

    def get_teacher(self):
        return self.teacher[0]

    def use_teacher_image(self):
        def teacher_image_encoder_fn(image, normalized=False):
            teacher = self.get_teacher()
            with torch.no_grad():
                return teacher.encode_image(image, normalized=normalized)

        self._image_encoder = FNBlock(teacher_image_encoder_fn)

        class EmptyVisual(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = 0
        self._image_encoder.visual = EmptyVisual()
        self._without_ddp[0] = self._image_encoder

    def use_teacher_text(self):
        def teacher_text_encoder_fn(text, normalized=False):
            teacher = self.get_teacher()
            with torch.no_grad():
                return teacher.encode_text(text, normalized=normalized)
        self._text_encoder = FNBlock(teacher_text_encoder_fn)

        class EmptyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = 0
        self._text_encoder.transformer = EmptyTransformer()
        self._text_encoder.token_embedding = None
        self._without_ddp[1] = self._text_encoder

    def ddpify(self, ddp_fn):
        def _ddp_fn(module):
            cnt = sum([p.numel()
                      for p in module.parameters() if p.requires_grad])
            if cnt > 0:
                return ddp_fn(module)
            return FakeDDP(module)
        self._image_encoder = _ddp_fn(self.image_encoder_without_ddp)
        self._text_encoder = _ddp_fn(self.text_encoder_without_ddp)
        self._logit_scale = _ddp_fn(self.logit_scale_without_ddp)

        self.used_ddp = True

    def forward(self, image, text, normalized=True):
        image_features = text_features = None
        if image is not None:
            with self.image_autocast():
                image_features = self._image_encoder(
                    image, normalized=normalized)
        if text is not None:
            with self.text_autocast():
                text_features = self._text_encoder(text, normalized=normalized)
        with self.logit_autocast():
            logit_scale = self._logit_scale(torch.tensor(0))
        return image_features, text_features, logit_scale.exp()

    def encode_image(self, image, normalized=False):
        with self.image_autocast():
            return self._image_encoder(image, normalized=normalized)

    def encode_text(self, text, normalized=False):
        with self.text_autocast():
            return self._text_encoder(text, normalized=normalized)

    @property
    def logit_scale(self):
        return self.logit_scale_without_ddp.logit_scale

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        tower = self.image_encoder_without_ddp
        for param in tower.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(tower)

    def lock_text_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        tower = self.text_encoder_without_ddp
        for param in tower.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(tower)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        visual = self.image_encoder_without_ddp.visual
        transformer = self.text_encoder_without_ddp.transformer
        if hasattr(visual, 'set_grad_checkpointing'):
            visual.set_grad_checkpointing(enable)
        if transformer is not None and hasattr(transformer, 'set_grad_checkpointing'):
            transformer.set_grad_checkpointing(enable)

    def image_named_params(self):
        return self._image_encoder.named_parameters()

    def text_named_params(self):
        return self._text_encoder.named_parameters()

    def joint_named_params(self):
        return self._logit_scale.named_parameters()

    def load_state_dict(self, state_dict, strict=True):
        state_dict = convert_to_new_checkpoint(state_dict, self.used_ddp)
        if not any(k.startswith('_image_encoder') for k in state_dict.keys()):
            self.use_teacher_image()

        for m in ['module.', '']:
            flag = f'_image_encoder.{m}visual.model.head.0.weight'
            if flag in state_dict:
                # LN
                state_dict[f'_image_encoder.{m}visual.ln_post.weight'] = state_dict.pop(
                    f'_image_encoder.{m}visual.model.head.0.weight')
                state_dict[f'_image_encoder.{m}visual.ln_post.bias'] = state_dict.pop(
                    f'_image_encoder.{m}visual.model.head.0.bias')
                # FC
                state_dict[f'_image_encoder.{m}visual.proj'] = state_dict.pop(
                    f'_image_encoder.{m}visual.model.head.1.weight').T
        new_state_dict = state_dict.copy()
        for k, v in new_state_dict.items():
            if '.module' in k:
                state_dict[k.replace('.module', '')] = v
                state_dict.pop(k)
        super().load_state_dict(state_dict, strict=strict)


class CLIP(CLIPBase):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            mask_image: bool = False,
            mask_text: bool = False,
            sparsity_warmup: int = 1000,
            sparsity: float = 0.25,
            start_sparsity: float = 0.0,
    ):

        vision_ocfg = None
        text_ocfg = None

        if isinstance(vision_cfg, dict):
            vision_ocfg = vision_cfg.pop('configs', None)
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        if isinstance(text_cfg, dict):
            text_ocfg = text_cfg.pop('configs', None)
            text_cfg = CLIPTextCfg(**text_cfg)

        mask_cfg = Namespace()
        mask_cfg.sparsity_warmup = sparsity_warmup
        mask_cfg.sparsity = sparsity
        mask_cfg.start_sparsity = start_sparsity

        if vision_ocfg is None:
            image_encoder = ImageEncoder(embed_dim, vision_cfg, quick_gelu,
                                         l0_module_image=mask_image,
                                         mask_cfg=mask_cfg)

        if text_ocfg is None:
            text_encoder = TextEncoder(embed_dim, text_cfg, quick_gelu,
                                       l0_module_text=mask_text, mask_cfg=mask_cfg)

        super().__init__(image_encoder, text_encoder)


def convert_to_new_checkpoint(state_dict, used_ddp=False):
    if '_logit_scale.module.logit_scale' in state_dict:
        if not used_ddp:
            new_checkpoint = dict()
            for k, v in state_dict.items():
                sp = k.split('.')
                assert sp[1] == 'module', (sp, state_dict.keys())
                k = '.'.join(sp[:1] + sp[2:])
                new_checkpoint[k] = v
            state_dict = new_checkpoint
        return state_dict
    if '_logit_scale.logit_scale' in state_dict:
        if used_ddp:
            new_checkpoint = dict()
            for k, v in state_dict.items():
                sp = k.split('.')
                k = '.'.join(sp[:1] + ['module'] + sp[1:])
                new_checkpoint[k] = v
            state_dict = new_checkpoint
        return state_dict
    image_prefix = '_image_encoder.'
    text_prefix = '_text_encoder.'
    logit_scale_prefix = '_logit_scale.'
    if used_ddp:
        image_prefix += 'module.'
        text_prefix += 'module.'
        logit_scale_prefix += 'module.'
    new_checkpoint = dict()
    if 'module.logit_scale' in state_dict:
        # remove the prefix module
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    if 'logit_scale' in state_dict:
        # old CLIP checkpoint
        for k, v in state_dict.items():
            if k.startswith('visual.'):
                new_checkpoint[image_prefix + k] = v
            elif k == 'logit_scale':
                new_checkpoint[logit_scale_prefix + 'logit_scale'] = v
            else:
                new_checkpoint[text_prefix + k] = v
    else:
        new_checkpoint = state_dict
    return new_checkpoint


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, )):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_from_openai_state_dict(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones(
        (batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    # FIXME detect different token configs (ie no class token, or more)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:
                                                 extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s',
                 old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


@torch.no_grad()
def load_pruned_model(model, pruned_state_dict):
    '''
    A full model loads the pruned state dict.

    Inputs:
        model_state_dict: the full model weights
        pruned_state_dict: the pruned model weights
    '''
    def _copy_to_full_weight(dst, src):
        assert dst.ndim == src.ndim, (dst.ndim, src.ndim)
        dst.zero_()
        dims = src.shape
        if len(dims) == 0:
            dst.copy_(src)
        else:
            slices = [slice(0, d) for d in dims]
            dst[slices].copy_(src)

    lambda_init_value = 10.0
    model_state_dict = model.state_dict()
    head_dim = model.transformer.head_dim

    pruned_state_dict = {k.replace('image_encoder_without_ddp', '_image_encoder').
                         replace('text_encoder_without_ddp', '_text_encoder'): v for k, v in pruned_state_dict.items()}

    for name, dst in model_state_dict.items():
        # auto weight inheritance model weight prefix
        dst_shape = dst.shape

        # copy weights
        if name in pruned_state_dict:
            src = pruned_state_dict[name]
            if 'attn.in_proj_weight' in name:
                # reshape: (3 * num_heads * head_dim, embed_dim) -> (3, num_heads, head_dim, embed_dim)
                assert len(src.shape) == 2
                _copy_to_full_weight(dst.view(3, -1, head_dim, dst_shape[-1]),
                                     src.view(3, -1, head_dim, src.shape[-1]))
            elif 'attn.in_proj_bias' in name:
                # reshape: (3 * num_heads * head_dim,) -> (3, num_heads, head_dim)
                assert len(src.shape) == 1
                _copy_to_full_weight(dst.view(3, -1, head_dim),
                                     src.view(3, -1, head_dim))
            else:
                _copy_to_full_weight(dst, src)
        else:
            if '.resblocks.' in name:
                # the layer has been pruned.
                dst.zero_()

    model_state_dict['_logit_scale.logit_scale'] = pruned_state_dict['_logit_scale.logit_scale']

    # prune hidden dimensions
    encoder_names = ['_image_encoder', '_text_encoder']
    hidden_size_img = pruned_state_dict['_image_encoder.visual.ln_pre.weight'].shape[0]
    hidden_size_txt = pruned_state_dict['_text_encoder.positional_embedding'].shape[1]
    hidden_sizes = [hidden_size_img, hidden_size_txt]

    for ename, hidden_size in zip(encoder_names, hidden_sizes):
        # reset lambda in l0 module
        model_state_dict[f'{ename}.l0_module.lambda_1'].fill_(
            lambda_init_value)
        model_state_dict[f'{ename}.l0_module.lambda_2'].fill_(
            lambda_init_value)
        # prune the last dimensions
        model_state_dict[f'{ename}.l0_module.hidden_loga'][hidden_size:].fill_(
            -lambda_init_value)

    def _get_layer_id(name):
        return int(name.split('resblocks.')[1].split('.')[0])

    for ename in encoder_names:
        # get the depth of the encoder
        encoder_keys = list(k for k in model_state_dict.keys() if ename in k)
        encoder_depth = max(_get_layer_id(k)
                            for k in encoder_keys if 'resblocks' in k) + 1
        pruned_encoder_keys = list(
            k for k in pruned_state_dict.keys() if ename in k)
        in_proj_weight_shapes = [None for _ in range(encoder_depth)]
        mlp_c_fc_shapes = [None for _ in range(encoder_depth)]
        for k in pruned_encoder_keys:
            if 'in_proj_weight' in k:
                d = _get_layer_id(k)
                in_proj_weight_shapes[d] = pruned_state_dict[k].shape
            elif 'mlp.c_fc.weight' in k:
                d = _get_layer_id(k)
                mlp_c_fc_shapes[d] = pruned_state_dict[k].shape

        for d in range(encoder_depth):
            # set heads_loga
            if in_proj_weight_shapes[d] is not None:
                num_heads = in_proj_weight_shapes[d][0] // head_dim // 3
                model_state_dict[f'{ename}.l0_module.heads_loga'][d,
                                                                  num_heads:].fill_(-lambda_init_value)
            else:
                # all heads have been pruned
                model_state_dict[f'{ename}.l0_module.heads_loga'][d,
                                                                  :].fill_(-lambda_init_value)

            # set intermediate_loga
            if mlp_c_fc_shapes[d] is not None:
                inter_size = mlp_c_fc_shapes[d][0]
                model_state_dict[f'{ename}.l0_module.intermediate_loga'][d,
                                                                         inter_size:].fill_(-lambda_init_value)
            else:
                # all intermediate dimensions have been pruned
                model_state_dict[f'{ename}.l0_module.intermediate_loga'][d,
                                                                         :].fill_(-lambda_init_value)

    model.load_state_dict(model_state_dict, strict=True)
