import os
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders import custom_transforms as tr


def twoTrainSeg(args, root):
    images_base = os.path.join(root, 'leftImg8bit', 'train')
    train_files = [os.path.join(looproot, filename) for looproot, _, filenames in os.walk(images_base)
                   for filename in filenames if filename.endswith('.png')]
    number_images = len(train_files)
    permuted_indices_ls = np.random.permutation(number_images)
    indices_1 = permuted_indices_ls[: int(0.5 * number_images) + 1]
    indices_2 = permuted_indices_ls[int(0.5 * number_images):]
    if len(indices_1) % 2 != 0 or len(indices_2) % 2 != 0:
        raise Exception('indices lists need to be even numbers for batch norm')
    return CityscapesSegmentation(args, split='train', indices_for_split=indices_1), CityscapesSegmentation(args,
                                                                                                            split='train',
                                                                                                            indices_for_split=indices_2)


class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    def __init__(self, args, root, split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.crop = self.args.crop_size
        if split.startswith('re'):
            self.images_base = os.path.join(self.root, 'leftImg8bit', self.split[2:])
            self.annotations_base = os.path.join(self.root, 'gtFine', self.split[2:])
        else:
            self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
            self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
        return self.transform(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def get_transform(self):
        if self.split == 'train':
            return tr.transform_tr(self.args, self.mean, self.std)
        elif self.split == 'val':
            return tr.transform_val(self.args, self.mean, self.std)
        elif self.split == 'test':
            return tr.transform_ts(self.args, self.mean, self.std)
        elif self.split == 'retrain':
            return tr.transform_retr(self.args, self.mean, self.std)
        elif self.split == 'reval':
            return tr.transform_reval(self.args, self.mean, self.std)


if __name__ == '__main__':
    from dataloaders.dataloader_utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.resize = 513
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='retrain')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
