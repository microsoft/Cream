#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data
The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import cv2
import os

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

# whether from the PNG are used or new colors are generated
generate_new_colors = True

json_file = './panoptic_cityscapes_mul2/panoptic/predictions.json'
segmentations_folder = './panoptic_cityscapes_mul2/panoptic/predictions/'
img_folder = '/home2/hongyuan/data/cityscapes/leftImg8bit/val/'
panoptic_coco_categories = './panoptic_coco_categories.json'
output_dir = 'cityscapes_vis_results'

os.makedirs(output_dir, exist_ok=True)

with open(json_file, 'r') as f:
    coco_d = json.load(f)

# ann = np.random.choice(coco_d['annotations'])

with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}

# find input img that correspond to the annotation
img = None
# for image_info in coco_d['images']:
for image_info in coco_d['images']:
    for ann in coco_d['annotations']:
        if image_info['id'] == ann['image_id']:
            try:
                img = np.array(
                    Image.open(os.path.join(img_folder, image_info['file_name'].split('_')[0], image_info['file_name'].split('gtFine_leftImg8bit.png')[0]+'leftImg8bit.png'))
                )
            except:
                print("Undable to find correspoding input image.")
            break

    segmentation = np.array(
        Image.open(os.path.join(segmentations_folder, ann['file_name'])),
        dtype=np.uint8
    )
    segmentation_id = rgb2id(segmentation)
    # find segments boundaries
    boundaries = find_boundaries(segmentation_id, mode='thick')

    if generate_new_colors:
        segmentation[:, :, :] = 0
        color_generator = IdGenerator(categegories)
        for segment_info in ann['segments_info']:
            try:
                color = color_generator.get_color(segment_info['category_id'])
                mask = segmentation_id == segment_info['id']
                segmentation[mask] = color
            except:
                pass

    # depict boundaries
    segmentation[boundaries] = [0, 0, 0]
    if img.shape[:2] == segmentation.shape[:2]:
        pass
    else:
        print('img: {} shape error! img shape: {} seg shape: {}'.format(ann['image_id'], img.shape[:2], segmentation.shape[:2]))
        continue
   
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        segmentation = cv2.addWeighted(img, 0.6, segmentation, 0.4, 0)
    except:
        import pdb; pdb.set_trace()
    cv2.imwrite(os.path.join(output_dir, '{}.jpg').format(ann['image_id']), img[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, '{}_mask.jpg').format(ann['image_id']), segmentation[:, :, ::-1])
    #if img is None:
    #    plt.figure()
    #    plt.imshow(segmentation)
    #    plt.axis('off')
    #else:
    #    plt.figure(figsize=(9, 5))
    #    plt.subplot(121)
    #    plt.imshow(img)
    #    plt.axis('off')
    #    plt.subplot(122)
    #    plt.imshow(segmentation)
    #    plt.axis('off')
    #    plt.tight_layout()
    #plt.show()
