# ------------------------------------------------------------------------------
# Loads Cityscapes semantic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import glob
import os

import numpy as np

from .base_dataset import BaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 2975,
                     'trainval': 3475,
                     'val': 500,
                     'test': 1525},
    num_classes=19,
    ignore_label=255,
)

_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}


class Cityscapes(BaseDataset):
    """
    Cityscapes semantic segmentation dataset.
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
                 **kwargs):
        super(Cityscapes, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale,
                                         scale_step_size, mean, std)

        self.num_classes = _CITYSCAPES_INFORMATION.num_classes
        self.ignore_label = _CITYSCAPES_INFORMATION.ignore_label
        self.label_pad_value = (self.ignore_label, )

        # Get image and annotation list.
        self.img_list = self._get_files('image', self.split)
        self.ann_list = self._get_files('label', self.split)

        assert len(self) == _CITYSCAPES_INFORMATION.splits_to_sizes[self.split]

        self.transform = build_transforms(self, is_train)

    def _get_files(self, data, dataset_split):
        """Gets files for the specified data type and dataset split.
        Args:
            data: String, desired data ('image' or 'label').
            dataset_split: String, dataset split ('train', 'val', 'test')
        Returns:
            A list of sorted file names or None when getting label for test set.
        """
        if data == 'label' and dataset_split == 'test':
            return None
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
        search_files = os.path.join(
            self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
        filenames = glob.glob(search_files)
        return sorted(filenames)

    @staticmethod
    def train_id_to_eval_id():
        return _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

    def _convert_train_id_to_eval_id(self, prediction):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(self.train_id_to_eval_id()):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap
