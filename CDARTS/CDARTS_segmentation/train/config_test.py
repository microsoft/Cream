# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.realpath(".")
C.root_dir = osp.realpath("..")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))

"""Data Dir"""
C.dataset_path = "/home/t-hongyuanyu/data/cityscapes/"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "cityscapes_train_fine.txt")
C.train_eval_source = osp.join(C.dataset_path, "cityscapes_train_val_fine.txt")
C.eval_source = osp.join(C.dataset_path, "cityscapes_val_fine.txt")
C.test_source = osp.join(C.dataset_path, "cityscapes_test.txt")

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))
add_path(C.root_dir)

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.target_size = 1024
C.down_sampling = 1 # first down_sampling then crop ......
C.gt_down_sampling = 1
C.num_train_imgs = 2975
C.num_eval_imgs = 500

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_stride_rate = 5 / 6
C.eval_scale_array = [1, ]
C.eval_flip = False
C.eval_base_size = 1024
C.eval_crop_size = 1024
C.eval_height = 1024
C.eval_width = 2048

C.layers = 16
C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
C.stem_head_width = (1, 1)
C.Fch = 20
C.image_height = 512
C.image_width = 1024

########################################
C.save = "test"
C.is_test = False # if True, prediction files for the test set will be generated
C.is_eval = True # if True, the train.py will only do evaluation for once
C.json_file = "./jsons/3path_big2.json"
C.model_path = "./3path_big2.pth.tar" # path to pretrained directory to be evaluated
