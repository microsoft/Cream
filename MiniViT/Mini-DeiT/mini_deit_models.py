import torch
from timm.models.registry import register_model
from models import deit_tiny_patch16_224,\
    deit_small_patch16_224,\
    deit_base_patch16_224,\
    deit_base_patch16_384


def get_deit_rpe_config():
    from irpe import get_rpe_config as _get_rpe_config
    rpe_config = _get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=0,
        rpe_on='k',
    )
    return rpe_config


@register_model
def mini_deit_tiny_patch16_224(pretrained=False, **kwargs):
    return deit_tiny_patch16_224(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times=2,
                                 use_transform=True,
                                 **kwargs)


@register_model
def mini_deit_small_patch16_224(pretrained=False, **kwargs):
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=False,
                                  repeated_times=2,
                                  use_transform=True,
                                  **kwargs)


@register_model
def mini_deit_base_patch16_224(pretrained=False, **kwargs):
    return deit_base_patch16_224(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times=2,
                                 use_transform=True,
                                 **kwargs)


@register_model
def mini_deit_base_patch16_384(pretrained=False, **kwargs):
    return deit_base_patch16_384(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times=2,
                                 use_transform=True,
                                 **kwargs)
