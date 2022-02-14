from __future__ import division
import os
import sys
import time
import glob
import json
import yaml
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
from thop import profile

from config_test import config
if config.is_eval:
    config.save = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
else:
    config.save = 'train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
from dataloader import get_train_loader, CyclicIterator
from datasets import Cityscapes

import dataloaders
from utils.init_func import init_weight
from eval import SegEvaluator
from test import SegTester

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_seg import Network_Multi_Path_Infer_SPOS as Network
import seg_metrics

from utils.pyt_utils import load_pretrain

## dist train
try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='../configs/auto2/sz512drop0.2.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--seed", default=12345, type=int)

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

    # dist init
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:26442', world_size=1, rank=0)
    config.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    logging.info("rank: {} world_size: {}".format(args.local_rank, args.world_size))
    
    if args.local_rank == 0:
        create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
        logger = SummaryWriter(config.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("args = %s", str(config))
    else:
        logger = None

    # preparation ################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # data loader ###########################
    if config.is_test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}
    else:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}

    with open(config.json_file, 'r') as f:
        model_dict = json.loads(f.read())

    model = Network(
        model_dict["ops"], model_dict["paths"], model_dict["downs"], model_dict["widths"], model_dict["lasts"],
        num_classes=config.num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width)

    if args.local_rank == 0:
        logging.info("net: " + str(model))
        flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
        logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))
        with open(os.path.join(config.save, 'args.yaml'), 'w') as f:
            f.write(args_text)

    model = model.cuda()
    init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    model = load_pretrain(model, config.model_path)
    
    # partial = torch.load(config.model_path)
    # state = model.state_dict()
    # pretrained_dict = {k: v for k, v in partial.items() if k in state}
    # state.update(pretrained_dict)
    # model.load_state_dict(state)

    eval_model = model
    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                config.image_std, eval_model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                verbose=False, save_path=None, show_image=False, show_prediction=False)
    tester = SegTester(Cityscapes(data_setting, 'test', None), config.num_classes, config.image_mean,
                                config.image_std, eval_model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                verbose=False, save_path=None, show_prediction=False)

    # Cityscapes ###########################################
    logging.info(config.model_path)
    logging.info(config.save)
    with torch.no_grad():
        if config.is_test:
            # test
            print("[test...]")
            with torch.no_grad():
                test(0, model, tester, logger)
        else:
            # validation
            print("[validation...]")
            valid_mIoU = infer(model, evaluator, logger)
            logger.add_scalar("mIoU/val", valid_mIoU, 0)
            logging.info("Model valid_mIoU %.3f"%(valid_mIoU))

def infer(model, evaluator, logger):
    model.eval()
    # _, mIoU = evaluator.run_online()
    _, mIoU = evaluator.run_online_multiprocess()
    return mIoU

def test(epoch, model, tester, logger):
    output_path = os.path.realpath('.')
    os.system("mkdir %s"%os.path.join(output_path, config.save, "test"))
    model.eval()
    tester.run_online_multiprocess()
    os.system("mv %s %s"%(os.path.join(output_path, config.save, "test"), os.path.join(output_path, config.save, "test_%d_%d"%(0, epoch))))

if __name__ == '__main__':
    main() 
