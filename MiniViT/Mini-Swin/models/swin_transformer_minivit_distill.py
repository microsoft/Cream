import torch
from torch.functional import norm
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .swin_transformer_minivit import Mlp, window_partition, window_reverse, PatchEmbed, PatchMerging




class WindowAttentionDISTILL(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            

    def forward(self, x, mask=None, proj_l=None, proj_w=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_tmp = self.qkv(x)  # (3, B_, num_heads, N, D)
        qkv = qkv_tmp.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (B_, num_heads, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_out = qkv_tmp.reshape(B_, N, 3, C).permute(2, 0, 1, 3)
        q = q * self.scale
        # (B_, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if proj_l is not None:
            attn = proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        if proj_w is not None:
            attn = proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, (qkv_out[0], qkv_out[1], qkv_out[2])

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlockDISTILL(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., shift_size=0., drop_path=[0],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 # The following arguments are for MiniViT
                 is_init_window_shift=False,
                 is_sep_layernorm = False,
                 is_transform_FFN=False,
                 is_transform_heads = False,):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.share_num = len(drop_path)
        self.is_init_window_shift = is_init_window_shift
        self.is_sep_layernorm = is_sep_layernorm
        self.is_transform_FFN = is_transform_FFN
        self.is_transform_heads = is_transform_heads

        if self.is_sep_layernorm:  
            self.norm1_list = nn.ModuleList()
            for _ in range(self.share_num):
                self.norm1_list.append(norm_layer(dim))
        else:
            self.norm1 = norm_layer(dim)

        if self.is_transform_heads:
            self.proj_l = nn.ModuleList()
            self.proj_w = nn.ModuleList()
            for _ in range(self.share_num):
                self.proj_l.append(nn.Linear(num_heads, num_heads))
                self.proj_w.append(nn.Linear(num_heads, num_heads))
        else:
            self.proj_l = None
            self.proj_w = None

        self.attn = WindowAttentionDISTILL(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= shift_size < self.window_size, "shift_size must in 0-window_size"
        self.shift_size = shift_size
        if shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.is_sep_layernorm:
            self.norm2_list = nn.ModuleList()
            for _ in range(self.share_num):
                self.norm2_list.append(norm_layer(dim))
        else:
            self.norm2 = norm_layer(dim)
        
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = nn.ModuleList()
        for index, drop_path_value in enumerate(drop_path):
            self.drop_path.append(DropPath(drop_path_value) if drop_path_value > 0. else nn.Identity())


        if self.is_transform_FFN:
            self.local_norm_list = nn.ModuleList()
            self.local_conv_list = nn.ModuleList()
            _window_size = 7
            for _ in range(self.share_num):
                self.local_norm_list.append(norm_layer(dim))
                self.local_conv_list.append(nn.Conv2d(dim, dim, _window_size, 1, _window_size // 2, groups=dim, bias=qkv_bias))
        else:
            self.local_conv_list = None
        

    def forward_feature(self, x, is_shift=False, layer_index=0):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if self.is_sep_layernorm:
            x = self.norm1_list[layer_index](x)
        else:
            x = self.norm1(x)

        x = x.view(B, H, W, C)

        # cyclic shift
        if is_shift and self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        proj_l = self.proj_l[layer_index] if self.is_transform_heads else self.proj_l
        proj_w = self.proj_w[layer_index] if self.is_transform_heads else self.proj_w

        # W-MSA/SW-MSA
        attn_windows, qkv_tuple = self.attn(x_windows, mask=self.attn_mask, proj_l=proj_l, proj_w=proj_w)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if is_shift and self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path[layer_index](x)

        if self.local_conv_list is not None:
            x = self.local_norm_list[layer_index](x)
            x = x.permute(0, 2, 1).view(B, C, H, W)
            x = x + self.local_conv_list[layer_index](x)
            x = x.view(B, C, H * W).permute(0, 2, 1)

        norm2 = self.norm2_list[layer_index] if self.is_sep_layernorm else self.norm2

        x = x + self.drop_path[layer_index](self.mlp(norm2(x)))
        return x, qkv_tuple

    def forward(self, x):
        init_window_shift = self.is_init_window_shift
        qkv_list = []
        hidden_list = []
        for index in range(self.share_num):
            x, qkv_tuple = self.forward_feature(x, init_window_shift, index)
            init_window_shift = not init_window_shift
            qkv_list.append(qkv_tuple)
            hidden_list.append(x)
        return x, qkv_list, hidden_list

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def custom_init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                print(f'Warning: {type(m)} uses default initialization...')

        for m in [self.proj_l, self.proj_w]:
            if m is not None:
                if isinstance(m, nn.ModuleList):
                    for layer in m:
                        layer.apply(_init_weights)
                else:
                    m.apply(_init_weights)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = self.window_size * self.window_size
        N_conv = 7 * 7
        nW = H * W / N
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops_attn = nW * self.attn.flops(N)
        # mlp
        flops_mlp = 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        # norm in transformation for MSA
        if self.is_transform_heads:
            flops_attn += nW * 2 * N * N * self.attn.num_heads * self.attn.num_heads
        # norm in transformation for FFN
        if self.is_transform_FFN:
            flops_mlp += nW * self.dim * N
            flops_mlp += nW * self.dim * N_conv * N
        # local 
        return flops + flops_attn + flops_mlp

class BasicLayerDISTILL(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=[0.], norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 # The following parameters are for MiniViT
                 is_sep_layernorm = False,
                 is_transform_FFN=False,
                 is_transform_heads = False,
                 separate_layer_num = 1,
                 ):

        super().__init__()
        ## drop path must be a list
        assert(isinstance(drop_path, list))

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.share_times = depth // separate_layer_num
        self.separate_layer_num = separate_layer_num

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(self.separate_layer_num):
            drop_path_list = drop_path[(i*self.share_times): min((i+1)*self.share_times, depth)]
            self.blocks.append(SwinTransformerBlockDISTILL(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path_list,
                                 norm_layer=norm_layer,
                                 ## The following arguments are for MiniViT
                                 is_init_window_shift = (i*self.share_times)%2==1,
                                 is_sep_layernorm = is_sep_layernorm,
                                 is_transform_FFN = is_transform_FFN,
                                 is_transform_heads = is_transform_heads,
                                ))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        qkv_tuple_list = []
        hidden_tuple_list = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, qkv_tuple_list_tmp, hidden_tuple_list_tmp = checkpoint.checkpoint(blk, x)
            else:
                x, qkv_tuple_list_tmp, hidden_tuple_list_tmp = blk(x)
            qkv_tuple_list += qkv_tuple_list_tmp
            hidden_tuple_list += hidden_tuple_list_tmp
        if self.downsample is not None:
            x = self.downsample(x)
        return x, qkv_tuple_list, hidden_tuple_list

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        flops += self.blocks[0].flops() * self.depth
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformerMiniViTDistill(nn.Module):
    r""" MiniViT for Swin Transformer
         The model structure is the same as SwinTransformerMiniViT.
         It will return extra outputs of self-attention and hidden state.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, 
                 ## The following arguments are for MiniViT
                 is_sep_layernorm = False,
                 is_transform_FFN = False,
                 is_transform_heads = False,
                 separate_layer_num_list = [1, 1, 2, 1],
                 is_student = False,
                 fit_size_C = 128,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.is_sep_layernorm=is_sep_layernorm
        self.is_transform_FFN=is_transform_FFN
        self.is_transform_heads=is_transform_heads
        self.separate_layer_num_list=separate_layer_num_list

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print("Dropout path: ", dpr)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerDISTILL(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               # The following arguments are for MiniviT
                               is_sep_layernorm = is_sep_layernorm,
                                is_transform_FFN = is_transform_FFN,
                                is_transform_heads = is_transform_heads,
                                separate_layer_num = separate_layer_num_list[i_layer],
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.is_student = is_student
        self.fit_size_C = fit_size_C
        self.fit_dense_C = nn.ModuleList()
        if self.is_student:
            for i_layer in range(self.num_layers):
                self.fit_dense_C.append(nn.Linear(int(embed_dim * 2 ** i_layer), int(fit_size_C* 2 ** i_layer)))

        self.apply(self._init_weights)
        self.apply(self._custom_init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _custom_init_weights(self, m):
        if hasattr(m, 'custom_init_weights'):
            m.custom_init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, layer_id_list=[], is_hidden_org=True):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        layer_id = 0
        qkv_tuple_return_list = []
        hidden_tuple_return_list = []

        for id, layer in enumerate(self.layers):
            x, qkv_tuple_list, hidden_tuple_list = layer(x)
            for index in range(len(qkv_tuple_list)):
                if index + layer_id in layer_id_list:
                    qkv_tuple_return_list.append(qkv_tuple_list[index])
                    if self.is_student and not is_hidden_org:
                        hidden_tuple_return_list.append(self.fit_dense_C[id](hidden_tuple_list[index]))
                    else:
                        hidden_tuple_return_list.append(hidden_tuple_list[index])
            layer_id += len(qkv_tuple_list)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, qkv_tuple_return_list, hidden_tuple_return_list

    def forward(self, x, layer_id_list=[], is_attn_loss=False, is_hidden_loss=False, is_hidden_org = True):
        is_hidden_rel = is_hidden_org

        x, qkv_tuple_return_list, hidden_tuple_return_list = self.forward_features(x, layer_id_list, is_hidden_org=is_hidden_rel)

        
        x = self.head(x)

        if is_attn_loss and is_hidden_loss:
            return x, qkv_tuple_return_list, hidden_tuple_return_list
        elif is_attn_loss:
            return x, qkv_tuple_return_list
        elif is_hidden_loss:
            return x, hidden_tuple_return_list
        else:
            return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
