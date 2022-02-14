import math
import torch
import random
import numpy as np
import torch.nn as nn
from numpy import int64 as int64
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        return {'image': img,
                'label': mask}


# random crop 321*321
class RandomCrop(object):
    def __init__(self, crop_size=320):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}


class RandomScale(object):
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        scale = random.choice(self.scales)
        w, h = int(w * scale), int(h * scale)
        return {'image': img,
                'label': mask}


class Retrain_Preprocess(object):
    def __init__(self, flip_prob, scale_range, crop, mean, std):
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.crop = crop
        self.data_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)])

    def __call__(self, sample):
        if self.flip_prob is not None and random.random() < self.flip_prob:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            sample['label'] = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)

        if self.scale_range is not None:
            w, h = sample['image'].size
            rand_log_scale = math.log(self.scale_range[0], 2) + random.random() * \
                (math.log(self.scale_range[1], 2) - math.log(self.scale_range[0], 2))
            random_scale = math.pow(2, rand_log_scale)
            new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
            sample['image'] = sample['image'].resize(new_size, Image.ANTIALIAS)
            sample['label'] = sample['label'].resize(new_size, Image.NEAREST)
        sample['image'] = self.data_transforms(sample['image'])
        sample['label'] = torch.LongTensor(np.array(sample['label']).astype(int64))

        if self.crop:
            image, mask = sample['image'], sample['label']
            h, w = image.shape[1], image.shape[2]
            pad_tb = max(0, self.crop[0] - h)
            pad_lr = max(0, self.crop[1] - w)
            image = nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
            mask = nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

            h, w = image.shape[1], image.shape[2]
            i = random.randint(0, h - self.crop[0])
            j = random.randint(0, w - self.crop[1])
            sample['image'] = image[:, i:i + self.crop[0], j:j + self.crop[1]]
            sample['label'] = mask[i:i + self.crop[0], j:j + self.crop[1]]
        return sample


class transform_tr(object):
    def __init__(self, args, mean, std):
        if args.multi_scale is None:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=args.resize),
                RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                Normalize(mean, std),
                ToTensor()])
        else:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=args.resize),
                RandomScale(scales=args.multi_scale),
                RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                Normalize(mean, std),
                ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_val(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.resize),
            FixScaleCrop(crop_size=args.crop_size),  # TODO:CHECK THIS
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_val(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.crop_size),
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_ts(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.crop_size),
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_retr(object):
    def __init__(self, args, mean, std):
        crop_size = (args.crop_size, args.crop_size) if isinstance(args.crop_size, int) else args.crop_size
        self.composed_transforms = Retrain_Preprocess(0.5, (0.5, 2), crop_size, mean, std)

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_reval(object):  # we use multi_scale evaluate in evaluate.py so dont need resize in dataset
    def __init__(self, args, mean, std):
        self.composed_transforms = Retrain_Preprocess(None, None, None, mean, std)

    def __call__(self, sample):
        return self.composed_transforms(sample)
