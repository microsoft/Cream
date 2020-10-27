# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

__C = CN()

cfg = __C

__C.AUTO_RESUME = True
__C.DATA_DIR = './data/imagenet'
__C.JOB_NAME = 'cream'
__C.RESUME_PATH = './experiments/ckps/resume.pth.tar'
__C.SAVE_PATH = './experiments/ckps/'
__C.SEED = 42
__C.LOG_INTERVAL = 50
__C.RECOVERY_INTERVAL = 0
__C.WORKERS = 4
__C.NUM_GPU = 1
__C.SAVE_IMAGES = False
__C.AMP = False
__C.OUTPUT = 'output/path/'
__C.EVAL_METRICS = 'prec1'
__C.TTA = 0  # Test/inference time augmentation
__C.LOCAL_RANK = 0
__C.VERBOSE = False

# dataset configs
__C.DATASET = CN()
__C.DATASET.NUM_CLASSES = 1000
__C.DATASET.IMAGE_SIZE = 224  # image patch size
__C.DATASET.INTERPOLATION = 'bilinear'  # Image resize interpolation type
__C.DATASET.BATCH_SIZE = 32  # batch size
__C.DATASET.NO_PREFECHTER = False
__C.DATASET.PIN_MEM = True
__C.DATASET.VAL_BATCH_MUL = 4


# model configs
__C.MODEL = CN()
__C.MODEL.SELECTION = 14
__C.MODEL.GP = 'avg'  # type of global pool ["avg", "max", "avgmax", "avgmaxc"]
__C.MODEL.DROPOUT_RATE = 0.0  # dropout rate

# model ema parameters
__C.MODEL.EMA = CN()
__C.MODEL.EMA.USE = True
__C.MODEL.EMA.FORCE_CPU = False  # force model ema to be tracked on CPU
__C.MODEL.EMA.DECAY = 0.9998

# optimizer configs
__C.OPTIMIZER = CN()
__C.OPTIMIZER.NAME = 'sgd'
__C.OPTIMIZER.MOMENTUM = 0.9
__C.OPTIMIZER.WEIGHT_DECAY = 1e-3

# scheduler configs
__C.SCHEDULER = CN()
__C.SCHEDULER.NAME = 'step'
__C.SCHEDULER.LR = 1e-2
__C.SCHEDULER.WARMUP_LR = 1e-4
__C.SCHEDULER.MIN_LR = 1e-5
__C.SCHEDULER.EPOCHS = 200
__C.SCHEDULER.WARMUP_EPOCHS = 3
__C.SCHEDULER.DECAY_RATE = 0.1

# data augmentation parameters
__C.AUGMENTATION = CN()
__C.AUGMENTATION.AA = 'rand-m9-mstd0.5'
__C.AUGMENTATION.COLOR_JITTER = 0.4
__C.AUGMENTATION.RE_PROB = 0.2  # random erase prob
__C.AUGMENTATION.RE_MODE = 'pixel'  # random erase mode
__C.AUGMENTATION.MIXUP = 0.0  # mixup alpha
__C.AUGMENTATION.MIXUP_OFF_EPOCH = 0  # turn off mixup after this epoch
__C.AUGMENTATION.SMOOTHING = 0.1  # label smoothing parameters

# batch norm parameters (only works with gen_efficientnet based models
# currently)
__C.BATCHNORM = CN()
__C.BATCHNORM.SYNC_BN = True
__C.BATCHNORM.BN_TF = False
__C.BATCHNORM.BN_MOMENTUM = 0.1  # batchnorm momentum override
__C.BATCHNORM.BN_EPS = 1e-5  # batchnorm eps override

# supernet training hyperparameters
__C.SUPERNET = CN()
__C.SUPERNET.UPDATE_ITER = 1300
__C.SUPERNET.SLICE = 4
__C.SUPERNET.POOL_SIZE = 10
__C.SUPERNET.RESUNIT = False
__C.SUPERNET.DIL_CONV = False
__C.SUPERNET.TINY = False
__C.SUPERNET.UPDATE_2ND = True
__C.SUPERNET.FLOPS_MAXIMUM = 600
__C.SUPERNET.FLOPS_MINIMUM = 0
__C.SUPERNET.PICK_METHOD = 'meta'  # pick teacher method
__C.SUPERNET.META_LR = 1e-4
__C.SUPERNET.META_STA_EPOCH = 20  # start using meta picking method
__C.SUPERNET.HOW_TO_PROB = 'pre_prob'  # sample method
__C.SUPERNET.PRE_PROB = (0.05, 0.2, 0.05, 0.5, 0.05,
                         0.15)  # sample prob in 'pre_prob'
