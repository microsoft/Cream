# --------------------------------------------------------
# TinyViT Model Builder
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------

from .tiny_vit import TinyViT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'tiny_vit':
        M = config.MODEL.TINY_VIT
        model = TinyViT(img_size=config.DATA.IMG_SIZE,
                        in_chans=M.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dims=M.EMBED_DIMS,
                        depths=M.DEPTHS,
                        num_heads=M.NUM_HEADS,
                        window_sizes=M.WINDOW_SIZES,
                        mlp_ratio=M.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        mbconv_expand_ratio=M.MBCONV_EXPAND_RATIO,
                        local_conv_size=M.LOCAL_CONV_SIZE,
                        layer_lr_decay=config.TRAIN.LAYER_LR_DECAY,
                        )
    elif model_type == 'clip_vit_large14_224':
        from .clip import CLIP
        kwargs = {
            'embed_dim': 768, 'image_resolution': 224,
            'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14,
            "num_classes": config.MODEL.NUM_CLASSES,
        }
        model = CLIP(**kwargs)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
