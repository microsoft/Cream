"""The implementation of models with image RPE"""
import torch
from timm.models.registry import register_model
from irpe import get_rpe_config
from models import deit_tiny_patch16_224,\
    deit_small_patch16_224,\
    deit_base_patch16_224


_checkpoint_url_prefix = \
    'https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/'
_provided_checkpoints = set([
    'deit_tiny_patch16_224_ctx_product_50_shared_k',
    'deit_small_patch16_224_ctx_product_50_shared_k',
    'deit_small_patch16_224_ctx_product_50_shared_qk',
    'deit_small_patch16_224_ctx_product_50_shared_qkv',
    'deit_base_patch16_224_ctx_product_50_shared_k',
    'deit_base_patch16_224_ctx_product_50_shared_qkv',
])


def register_rpe_model(fn):
    '''Register a model with iRPE
    It is a wrapper of `register_model` with loading the pretrained checkpoint.
    '''
    def fn_wrapper(pretrained=False, **kwargs):
        model = fn(pretrained=False, **kwargs)
        if pretrained:
            model_name = fn.__name__
            assert model_name in _provided_checkpoints, \
                f'Sorry that the checkpoint `{model_name}` is not provided yet.'
            url = _checkpoint_url_prefix + model_name + '.pth'
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url,
                map_location='cpu', check_hash=False,
            )
            model.load_state_dict(checkpoint['model'])

        return model

    # rename the name of fn_wrapper
    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


##### DeiT-Tiny with image relative position encoding

@register_rpe_model
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

@register_rpe_model
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


@register_rpe_model
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


@register_rpe_model
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


@register_rpe_model
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


@register_rpe_model
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


@register_rpe_model
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

@register_rpe_model
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


@register_rpe_model
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
