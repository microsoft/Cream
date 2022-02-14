# ------------------------------------------------------------------------------
# Data augmentation following DeepLab
# (https://github.com/tensorflow/models/blob/master/research/deeplab/input_preprocess.py#L28).
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import random

import cv2
import numpy as np
from torchvision.transforms import functional as F


class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """
    Converts image to torch Tensor.
    """
    def __call__(self, image, label):
        return F.to_tensor(image), label


class Normalize(object):
    """
    Normalizes image by mean and std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


class RandomScale(object):
    """
    Applies random scale augmentation.
    Arguments:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        scale_step_size: The step size from minimum to maximum value.
    """
    def __init__(self, min_scale, max_scale, scale_step_size):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

    @staticmethod
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Gets a random scale value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        Raises:
            ValueError: min_scale_factor has unexpected value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    def __call__(self, image, label):
        f_scale = self.get_random_scale(self.min_scale, self.max_scale, self.scale_step_size)
        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
        image_dtype = image.dtype
        label_dtype = label.dtype
        image = cv2.resize(image.astype(np.float), None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float), None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image.astype(image_dtype), label.astype(label_dtype)


class RandomCrop(object):
    """
    Applies random crop augmentation.
    Arguments:
        crop_h: Integer, crop height size.
        crop_w: Integer, crop width size.
        pad_value: Tuple, pad value for image, length 3.
        ignore_label: Tuple, pad value for label, length could be 1 (semantic) or 3 (panoptic).
        random_pad: Bool, when crop size larger than image size, whether to randomly pad four boundaries,
            or put image to top-left and only pad bottom and right boundaries.
    """
    def __init__(self, crop_h, crop_w, pad_value, ignore_label, random_pad):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad

    def __call__(self, image, label):
        img_h, img_w = image.shape[0], image.shape[1]
        # save dtype
        image_dtype = image.dtype
        label_dtype = label.dtype
        # padding
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            img_pad = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                         value=self.pad_value)
            label_pad = cv2.copyMakeBorder(label, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        else:
            img_pad, label_pad = image, label
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        return image.astype(image_dtype), label.astype(label_dtype)


class RandomHorizontalFlip(object):
    """
    Applies random flip augmentation.
    Arguments:
        prob: Probability of flip.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
            image = image[:, ::-1].copy()
            label = label[:, ::-1].copy()
        return image, label
