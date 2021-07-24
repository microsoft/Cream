"""The implementation of models with image RPE"""
from timm.models.registry import register_model
from irpe import get_rpe_config
from models import deit_tiny_patch16_224,\
    deit_small_patch16_224,\
    deit_base_patch16_224


##### DeiT-Tiny with image relative position encoding

@register_model
def deit_tiny_patch16_224_ctx_product_50_shared_k(pretrained=False, **kwargs):
    # DeiT-Tiny with relative position encoding on keys (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_tiny_patch16_224(pretrained=pretrained,
                                 rpe_config=rpe_config,
                                 **kwargs)


##### DeiT-Small with image relative position encoding

@register_model
def deit_small_patch16_224_ctx_euc_20_shared_k(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on keys (Contextual Euclidean method)
    rpe_config = get_rpe_config(
        ratio=20,
        method="euc",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


@register_model
def deit_small_patch16_224_ctx_quant_51_shared_k(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on keys (Contextual Quantization method)
    rpe_config = get_rpe_config(
        ratio=33,
        method="quant",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


@register_model
def deit_small_patch16_224_ctx_cross_56_shared_k(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on keys (Contextual Cross method)
    rpe_config = get_rpe_config(
        ratio=20,
        method="cross",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


@register_model
def deit_small_patch16_224_ctx_product_50_shared_k(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on keys (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


@register_model
def deit_small_patch16_224_ctx_product_50_shared_qk(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on queries and keys (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='qk',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


@register_model
def deit_small_patch16_224_ctx_product_50_shared_qkv(pretrained=False, **kwargs):
    # DeiT-Small with relative position encoding on queries, keys and values (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='qkv',
    )
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=rpe_config,
                                  **kwargs)


##### DeiT-Base with image relative position encoding

@register_model
def deit_base_patch16_224_ctx_product_50_shared_k(pretrained=False, **kwargs):
    # DeiT-Base with relative position encoding on keys (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='k',
    )
    return deit_base_patch16_224(pretrained=pretrained,
                                 rpe_config=rpe_config,
                                 **kwargs)


@register_model
def deit_base_patch16_224_ctx_product_50_shared_qkv(pretrained=False, **kwargs):
    # DeiT-Base with relative position encoding on queries, keys and values (Contextual Product method)
    rpe_config = get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='qkv',
    )
    return deit_base_patch16_224(pretrained=pretrained,
                                 rpe_config=rpe_config,
                                 **kwargs)


if __name__ == '__main__':
    import torch
    x = torch.randn(1, 3, 224, 224)
    model = deit_small_patch16_224_ctx_cross_50_shared_k()
    print(model)
    y = model(x)
    print(y.shape)
