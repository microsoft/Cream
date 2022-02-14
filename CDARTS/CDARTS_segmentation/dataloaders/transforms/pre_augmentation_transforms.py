# ------------------------------------------------------------------------------
# Builds transformation before data augmentation.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import warnings

import cv2
import math
import numpy as np


class Resize(object):
    """
    Applies random scale augmentation.
    Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/input_preprocess.py#L28
    Arguments:
        min_resize_value: Desired size of the smaller image side, no resize if set to None
        max_resize_value: Maximum allowed size of the larger image side, no limit if set to None
        resize_factor: Resized dimensions are multiple of factor plus one.
        keep_aspect_ratio: Boolean, keep aspect ratio or not. If True, the input
            will be resized while keeping the original aspect ratio. If False, the
            input will be resized to [max_resize_value, max_resize_value] without
            keeping the original aspect ratio.
        align_corners: If True, exactly align all 4 corners of input and output.
    """
    def __init__(self, min_resize_value=None, max_resize_value=None, resize_factor=None,
                 keep_aspect_ratio=True, align_corners=False):
        if min_resize_value is not None and min_resize_value < 0:
            min_resize_value = None
        if max_resize_value is not None and max_resize_value < 0:
            max_resize_value = None
        if resize_factor is not None and resize_factor < 0:
            resize_factor = None
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.keep_aspect_ratio = keep_aspect_ratio
        self.align_corners = align_corners

        if self.align_corners:
            warnings.warn('`align_corners = True` is not supported by opencv.')

        if self.max_resize_value is not None:
            # Modify the max_size to be a multiple of factor plus 1 and make sure the max dimension after resizing
            # is no larger than max_size.
            if self.resize_factor is not None:
                self.max_resize_value = (self.max_resize_value - (self.max_resize_value - 1) % self.resize_factor)

    def __call__(self, image, label):
        if self.min_resize_value is None:
            return image, label
        [orig_height, orig_width, _] = image.shape
        orig_min_size = np.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = self.min_resize_value / orig_min_size
        large_height = int(math.floor(orig_height * large_scale_factor))
        large_width = int(math.floor(orig_width * large_scale_factor))
        large_size = np.array([large_height, large_width])

        new_size = large_size
        if self.max_resize_value is not None:
            # Calculate the smaller of the possible sizes, use that if the larger is too big.
            orig_max_size = np.maximum(orig_height, orig_width)
            small_scale_factor = self.max_resize_value / orig_max_size
            small_height = int(math.floor(orig_height * small_scale_factor))
            small_width = int(math.floor(orig_width * small_scale_factor))
            small_size = np.array([small_height, small_width])

            if np.max(large_size) > self.max_resize_value:
                new_size = small_size

        # Ensure that both output sides are multiples of factor plus one.
        if self.resize_factor is not None:
            new_size += (self.resize_factor - (new_size - 1) % self.resize_factor) % self.resize_factor
            # If new_size exceeds largest allowed size
            new_size[new_size > self.max_resize_value] -= self.resize_factor

        if not self.keep_aspect_ratio:
            # If not keep the aspect ratio, we resize everything to max_size, allowing
            # us to do pre-processing without extra padding.
            new_size = [np.max(new_size), np.max(new_size)]

        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
        image_dtype = image.dtype
        label_dtype = label.dtype
        # cv2: (width, height)
        image = cv2.resize(image.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        return image.astype(image_dtype), label.astype(label_dtype)
