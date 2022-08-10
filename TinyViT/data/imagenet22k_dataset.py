# --------------------------------------------------------
# TinyViT ImageNet 22k Dataset
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------

import io
import os
import torch
from collections import defaultdict
from PIL import Image
import zipfile


class IN22KDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform, fname_format='{}.jpeg', debug=False):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.debug = debug
        self.fname_format = fname_format

        info_fname = os.path.join(data_root, 'in22k_image_names.txt')
        assert os.path.isfile(
            info_fname), f'IN22k-List filelist: {info_fname} does not exist'

        folders = defaultdict(list)
        with open(info_fname, 'r') as f:
            for iname in f:
                iname = iname.strip()
                class_name = iname[:iname.index('_')]
                folders[class_name].append(iname)
        class_names = sorted(folders.keys())
        self.nb_classes = len(class_names)

        if debug:
            for name in class_names:
                if not name.startswith('n00007846'):
                    folders[name] = []

        self.data = []
        for cls_id, cls_name in enumerate(class_names):
            self.data.extend([(iname, cls_id) for iname in folders[cls_name]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iname, target = self.data[idx]
        iob = self._read_file(iname)
        img = Image.open(iob).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _read_file(self, iname):
        # Example:
        # iname: 'n00007846_10001'
        # fname: 'n00007846_10001.jpeg'
        cls_name = iname[:iname.index('_')]
        fname = self.fname_format.format(iname)
        zip_fname = os.path.join(self.data_root, cls_name + '.zip')
        handle = zipfile.ZipFile(zip_fname, 'r')
        bstr = handle.read(fname)
        iob = io.BytesIO(bstr)
        return iob

    def get_keys(self):
        return [e[0] for e in self.data]


if __name__ == '__main__':
    data_root = './ImageNet-22k'
    def transform(x): return x
    fname_format = 'imagenet22k/{}.JPEG'
    dataset = IN22KDataset(data_root, transform, fname_format, debug=True)
    for img, target in dataset:
        print(type(img), target)
        break
