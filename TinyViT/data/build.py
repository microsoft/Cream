# --------------------------------------------------------
# TinyViT Data Builder
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted for TinyVIT
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import Mixup
from timm.data import create_transform

from .augmentation import create_transform as create_transform_record
from .augmentation.mixup import Mixup as Mixup_record
from .augmentation.dataset_wrapper import DatasetWrapper
from .imagenet22k_dataset import IN22KDataset
from .sampler import MyDistributedSampler

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config)
    config.freeze()

    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None

    sampler_train = MyDistributedSampler(
        dataset_train, shuffle=True,
        drop_last=False, padding=True, pair=mixup_active and config.DISTILL.ENABLED,
    )

    sampler_val = MyDistributedSampler(
        dataset_val, shuffle=False,
        drop_last=False, padding=False, pair=False,
    )

    # TinyViT Dataset Wrapper
    if config.DISTILL.ENABLED:
        dataset_train = DatasetWrapper(dataset_train,
                                       logits_path=config.DISTILL.TEACHER_LOGITS_PATH,
                                       topk=config.DISTILL.LOGITS_TOPK,
                                       write=config.DISTILL.SAVE_TEACHER_LOGITS,
                                       )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # modified for TinyViT, we save logits of all samples
        drop_last=not config.DISTILL.SAVE_TEACHER_LOGITS,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    if mixup_active:
        mixup_t = Mixup if not config.DISTILL.ENABLED else Mixup_record
        if config.DISTILL.ENABLED and config.AUG.MIXUP_MODE != "pair2":
            # change to pair2 mode for saving logits
            config.defrost()
            config.AUG.MIXUP_MODE = 'pair2'
            config.freeze()
        mixup_fn = mixup_t(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):

    transform = build_transform(is_train, config)
    dataset_tar_t = TimmDatasetTar

    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        # load tar dataset
        data_dir = os.path.join(config.DATA.DATA_PATH, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = dataset_tar_t(data_dir, transform=transform)
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22k':
        if is_train:
            dataset = IN22KDataset(data_root=config.DATA.DATA_PATH, transform=transform,
                                   fname_format=config.DATA.FNAME_FORMAT, debug=config.DATA.DEBUG)
            nb_classes = 21841
        else:
            # load ImageNet-1k validation set
            '''
            datasets/
            ├── ImageNet-22k/  # the folder of IN-22k
            └── ImageNet/  # the folder of IN-1k
            '''
            old_data_path = config.DATA.DATA_PATH
            config.defrost()
            config.DATA.DATA_PATH = os.path.normpath(
                os.path.join(old_data_path, '../ImageNet'))
            config.DATA.DATASET = 'imagenet'
            dataset, nb_classes = build_dataset(is_train=False, config=config)
            config.DATA.DATA_PATH = old_data_path
            config.DATA.DATASET = 'imagenet22k'
            config.freeze()
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32

    # RGB: mean, std
    rgbs = dict(
        default=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        inception=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
        clip=((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
    )
    mean, std = rgbs[config.DATA.MEAN_AND_STD_TYPE]

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        create_transform_t = create_transform if not config.DISTILL.ENABLED else create_transform_record
        transform = create_transform_t(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(
                    config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform
