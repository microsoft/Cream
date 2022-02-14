# ------------------------------------------------------------------------------
# Loads Cityscapes panoptic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import json
import os

import numpy as np

from .cityscapes import Cityscapes
from .utils import DatasetDescriptor
from ..transforms import build_transforms, PanopticTargetGenerator, SemanticTargetGenerator

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 2975,
                     'trainval': 3475,
                     'val': 500,
                     'test': 1525},
    num_classes=19,
    ignore_label=255,
)

# Add 1 void label.
_CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                            23, 24, 25, 26, 27, 28, 31, 32, 33, 0]

_CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]


class CityscapesPanoptic(Cityscapes):
    """
    Cityscapes panoptic segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
        semantic_only: Bool, only use semantic segmentation label.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
    """
    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(CityscapesPanoptic, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale,
                                                 scale_step_size, mean, std)

        self.num_classes = _CITYSCAPES_INFORMATION.num_classes
        self.ignore_label = _CITYSCAPES_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 1000
        self.label_dtype = np.float32
        self.thing_list = _CITYSCAPES_THING_LIST

        # Get image and annotation list.
        if split == 'test':
            self.img_list = self._get_files('image', self.split)
            self.ann_list = None
            self.ins_list = None
        else:
            self.img_list = []
            self.ann_list = []
            self.ins_list = []
            json_filename = os.path.join(self.root, 'gtFine', 'cityscapes_panoptic_{}_trainId.json'.format(self.split))
            dataset = json.load(open(json_filename))
            for img in dataset['images']:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(
                    self.root, 'leftImg8bit', self.split, img_file_name.split('_')[0],
                    img_file_name.replace('_gtFine', '')))
            for ann in dataset['annotations']:
                ann_file_name = ann['file_name']
                self.ann_list.append(os.path.join(
                    self.root, 'gtFine', 'cityscapes_panoptic_{}_trainId'.format(self.split), ann_file_name))
                self.ins_list.append(ann['segments_info'])

        assert len(self) == _CITYSCAPES_INFORMATION.splits_to_sizes[self.split]

        self.transform = build_transforms(self, is_train)
        if semantic_only:
            self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _CITYSCAPES_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    @staticmethod
    def train_id_to_eval_id():
        return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])
