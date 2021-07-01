# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

# This file is to demonstrate how to generate subImagenet.

import os
data_path = './data'
ImageNet_train_path = os.path.join(data_path, 'imagenet/train')
subImageNet_name = 'subImageNet'
class_idx_txt_path = os.path.join(data_path, subImageNet_name)

# train
classes = sorted(os.listdir(ImageNet_train_path))
if not os.path.exists(os.path.join(data_path, subImageNet_name)):
    os.mkdir(os.path.join(data_path, subImageNet_name))

subImageNet = dict()
with open(os.path.join(class_idx_txt_path, 'subimages_list.txt'), 'w') as f:
    subImageNet_class = classes[:100]
    for iclass in subImageNet_class:
        class_path = os.path.join(ImageNet_train_path, iclass)
        if not os.path.exists(
            os.path.join(
                data_path,
                subImageNet_name,
                iclass)):
            os.mkdir(os.path.join(data_path, subImageNet_name, iclass))
        images = sorted(os.listdir(class_path))
        subImages = images[:350]
        # print("{}\n".format(subImages))
        f.write("{}\n".format(subImages))
        subImageNet[iclass] = subImages
        for image in subImages:
            raw_path = os.path.join(ImageNet_train_path, iclass, image)
            new_ipath = os.path.join(
                data_path, subImageNet_name, iclass, image)
            os.system('cp {} {}'.format(raw_path, new_ipath))

sub_classes = sorted(subImageNet.keys())
with open(os.path.join(class_idx_txt_path, 'info.txt'), 'w') as f:
    class_idx = 0
    for key in sub_classes:
        images = sorted((subImageNet[key]))
        # print(len(images))
        f.write("{}\n".format(key))
        class_idx = class_idx + 1
