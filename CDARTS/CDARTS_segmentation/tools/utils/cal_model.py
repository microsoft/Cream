from __future__ import division
import os
import sys
import time
import glob
import json
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tensorboardX import SummaryWriter

import numpy as np
from thop import profile
from ptflops import get_model_complexity_info

from config_train import config
# if config.is_eval:
#     config.save = '../OUTPUT/eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
# else:
#     config.save = '../OUTPUT/train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
from dataloader import get_train_loader, CyclicIterator
from datasets import Cityscapes

import dataloaders
from utils.init_func import init_weight
from utils.lr_scheduler import Iter_LR_Scheduler
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator
from test import SegTester

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from utils.dist_utils import reduce_tensor, ModelEma
from model_seg import Network_Multi_Path_Infer_SPOS as Network
import seg_metrics

import yaml
import timm
from timm.optim import create_optimizer
from utils.pyt_utils import AverageMeter, to_cuda, get_loss_info_str, compute_hist, compute_hist_np, load_pretrain

def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='../configs/auto2/sz512drop0.2.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--det2_cfg', type=str, default='configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml', help='')
parser.add_argument('--save', type=str, default='../OUTPUT/train', help='')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--eval_height", default=1025, type=int, help='train height')
parser.add_argument("--eval_width", default=2049, type=int, help='train width')
parser.add_argument("--test_epoch", default=250, type=int, help='Epochs for test')
parser.add_argument("--batch_size", default=12, type=int, help='batch size')
parser.add_argument("--Fch", default=12, type=int, help='Fch')
parser.add_argument('--stem_head_width', type=float, default=1.0, help='base learning rate')
parser.add_argument('--resume', type=str, default='../OUTPUT/train/', help='resume')

## new retrain ###
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--epochs', type=int, default=4000, help='num of training epochs')
parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
parser.add_argument('--base_lr', type=float, default=0.05, help='base learning rate')
parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
parser.add_argument('--lr-step', type=float, default=None)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--min-lr', type=float, default=None)
parser.add_argument('--crop_size', type=int, default=769, help='image crop size')
parser.add_argument('--resize', type=int, default=769, help='image crop size')
parser.add_argument("--image_height", default=513, type=int, help='train height')
parser.add_argument("--image_width", default=1025, type=int, help='train width')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--dist', type=bool, default=True)
parser.add_argument('--autodeeplab', type=str, default='train_seg')
parser.add_argument('--max-iteration', default=1000000, type=bool)
parser.add_argument('--mode', default='poly', type=str, help='how lr decline')
parser.add_argument('--train_mode', type=str, default='iter', choices=['iter', 'epoch'])

parser.add_argument("--data_path", default='/home/t-hongyuanyu/data/cityscapes', type=str, help='If specified, replace config.load_path')
parser.add_argument("--load_path", default='', type=str, help='If specified, replace config.load_path')
parser.add_argument("--json_file", default='jsons/0.json', type=str, help='model_arch')
parser.add_argument("--seed", default=12345, type=int, help="random seed")
parser.add_argument('--sync_bn', action='store_false',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--random_sample', action='store_true',
                    help='Random sample path.')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path prob')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# train val
parser.add_argument('--ignore', type=int, default=255, help='semantic ignore')
parser.add_argument('--eval_flip', action='store_true', default=False,
                    help='semantic eval flip')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    
    
    if args.load_path:
        config.load_path = args.load_path

    config.batch_size = args.batch_size
    config.image_height = args.image_height
    config.image_width = args.image_width
    config.eval_height = args.eval_height
    config.eval_width = args.eval_width
    config.Fch = args.Fch
    config.dataset_path = args.data_path
    config.save = args.save

    # preparation ################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_files = glob.glob("Search/1paths/*.json") + glob.glob("Search/2paths/*.json") + glob.glob("Search/3paths/*.json")

    for model_file in model_files:

        with open(model_file, 'r') as f:
            # dict_a = json.loads(f, cls=NpEncoder)
            model_dict = json.loads(f.read())

        model = Network(
            model_dict["ops"], model_dict["paths"], model_dict["downs"], model_dict["widths"], model_dict["lasts"],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, stem_head_width=(args.stem_head_width, args.stem_head_width))

        if args.local_rank == 0:
            print("net: " + str(model))
            # with torch.cuda.device(0):
            #     macs, params = get_model_complexity_info(model, (3, 1024, 2048), as_strings=True,
            #                                     print_per_layer_stat=True, verbose=True)
            #     logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            #     logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

            flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
            flops = flops / 1e9
            params = params / 1e6
            model_dict['flops'] = flops
            model_dict['params'] = params
            print("params = %fMB, FLOPs = %fGB", params, flops)
        
        with open(model_file, 'w') as f:
                json.dump(model_dict, f, cls=NpEncoder)


if __name__ == '__main__':
    main()
    #launch(
    #    main,
    #    2,
    #    num_machines=1,
    #    machine_rank=0,
    #    dist_url='auto',
    #) 
