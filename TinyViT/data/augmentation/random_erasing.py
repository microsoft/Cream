""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman
"""
from .aug_random import random, np_random
import numpy as np
import math
import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if not per_pixel and not rand_color:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
    if per_pixel:
        shape = patch_size
    elif rand_color:
        shape = (patch_size[0], 1, 1)
    # normal_
    seed = random.randint(0, 1 << 30)
    bg = np.random.MT19937(seed)
    g = np.random.Generator(bg)
    x = g.normal(size=shape)
    return torch.tensor(x, dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """
    REF_H = 224
    REF_W = 224

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        ref_h, ref_w = self.REF_H, self.REF_W
        ref_area = ref_h * ref_w
        area = img_h * img_w
        for _ in range(count):
            for attempt in range(10):
                r1 = random.uniform(self.min_area, self.max_area)
                target_area = r1 * ref_area / count
                r2 = random.uniform(*self.log_aspect_ratio)
                aspect_ratio = math.exp(r2)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < ref_w and h < ref_h:
                    top = random.randint(0, ref_h - h)
                    left = random.randint(0, ref_w - w)
                    # ref -> now
                    top = min(int(round(top * img_h / ref_h)), img_h - 1)
                    left = min(int(round(left * img_w / ref_w)), img_w - 1)
                    h = min(int(round(h * img_h / ref_h)), img_h - top)
                    w = min(int(round(w * img_w / ref_w)), img_w - left)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs
