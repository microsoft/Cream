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
import _init_paths
from ptflops import get_model_complexity_info
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
from cydas import CyDASseg as Network
import seg_metrics

import yaml
import timm
from timm.optim import create_optimizer
from utils.pyt_utils import AverageMeter, to_cuda, get_loss_info_str, compute_hist, compute_hist_np, load_pretrain

from detectron2.config import get_cfg
from detectron2.engine import launch, default_setup, default_argument_parser
import detectron2.data.transforms as T
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)

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

def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='../configs/auto2/cydas.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--det2_cfg', type=str, default='configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml', help='')
parser.add_argument('--save', type=str, default='../OUTPUT/train_', help='')
parser.add_argument('--exp_name', type=str, default='cydas', help='')
parser.add_argument('--pretrain', type=str, default=None, help='resume path')
parser.add_argument('--size_divisibility', type=int, default=32, help='size_divisibility')
parser.add_argument('--resume', type=str, default='../OUTPUT/train/', help='resume path')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--eval_height", default=1025, type=int, help='train height')
parser.add_argument("--eval_width", default=2049, type=int, help='train width')
parser.add_argument("--test_epoch", default=250, type=int, help='Epochs for test')
parser.add_argument("--batch_size", default=12, type=int, help='batch size')
parser.add_argument("--Fch", default=12, type=int, help='Fch')
parser.add_argument('--stem_head_width', type=float, default=1.0, help='base learning rate')

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
parser.add_argument('--layers', type=int, default=20, help='layers')
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

parser.add_argument("--data_path", default='/home/hongyuan/data/cityscapes', type=str, help='If specified, replace config.load_path')
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
parser.add_argument('--bn_eps', type=float, default=1e-5, help='bn eps')
parser.add_argument('--bn_momentum', type=float, default=0.01, help='bn momentum')
parser.add_argument('--ignore', type=int, default=255, help='semantic ignore')
parser.add_argument('--eval_flip', action='store_true', default=False,
                    help='semantic eval flip')


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

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main():
    args, args_text = _parse_args()
    
    # detectron2 data loader ###########################
    # det2_args = default_argument_parser().parse_args()
    det2_args = args
    det2_args.config_file = args.det2_cfg
    cfg = setup(det2_args)
    mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
    det2_dataset = iter(build_detection_train_loader(cfg, mapper=mapper))
    
    # dist init
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    
    args.save = args.save + args.exp_name

    if args.local_rank == 0:
        create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
        logger = SummaryWriter(args.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("args = %s", str(args))
    else:
        logger = None

    # preparation ################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # config network and criterion ################
    gt_down_sampling = 1
    min_kept = int(args.batch_size * args.image_height * args.image_width // (16 * gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # data loader ###########################

    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    train_loader, train_sampler, val_loader, val_sampler, num_classes = dataloaders.make_data_loader(args, **kwargs)

    with open(args.json_file, 'r') as f:
        # dict_a = json.loads(f, cls=NpEncoder)
        model_dict = json.loads(f.read())

    width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    model = Network(Fch=args.Fch, num_classes=num_classes, stem_head_width=(args.stem_head_width, args.stem_head_width))

    last = model_dict["lasts"]

    if args.local_rank == 0:
        logging.info("net: " + str(model))
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (3, 1024, 2048), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
            logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        with open(os.path.join(args.save, 'args.yaml'), 'w') as f:
            f.write(args_text)

    init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, args.bn_eps, args.bn_momentum, mode='fan_in', nonlinearity='relu')

    if args.pretrain:
        model.backbone = load_pretrain(model.backbone, args.pretrain)
    model = model.cuda()

    # if args.sync_bn:
    #     if has_apex:
    #         model = apex.parallel.convert_syncbn_model(model)
    #     else:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Optimizer ###################################
    base_lr = args.base_lr

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = create_optimizer(args, model)
        
    if args.sched == "raw":
        lr_scheduler =None
    else:
        max_iteration = len(train_loader) * args.epochs
        lr_scheduler = Iter_LR_Scheduler(args, max_iteration, len(train_loader))

    start_epoch = 0
    if os.path.exists(os.path.join(args.save, 'last.pth.tar')):
        args.resume = os.path.join(args.save, 'last.pth.tar')

    if args.resume:
        model_state_file = args.resume
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
            start_epoch = checkpoint['start_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_epoch']))

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=None)

    if model_ema:
        eval_model = model_ema.ema
    else:
        eval_model = model

    if has_apex:
        model = DDP(model, delay_allreduce=True)
    else:
        model = DDP(model, device_ids=[args.local_rank])

    best_valid_iou = 0.
    best_epoch = 0

    logging.info("rank: {} world_size: {}".format(args.local_rank, args.world_size))
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            logging.info(args.load_path)
            logging.info(args.save)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        drop_prob = args.drop_path_prob * epoch / args.epochs
        # model.module.drop_path_prob(drop_prob)

        train_mIoU = train(train_loader, det2_dataset, model, model_ema, ohem_criterion, num_classes, lr_scheduler, optimizer, logger, epoch, args, cfg)

        torch.cuda.empty_cache()

        if epoch > args.epochs // 3:
        # if epoch >= 10:
            temp_iou, avg_loss = validation(val_loader, eval_model, ohem_criterion, num_classes, args, cal_miou=True)
        else:
            temp_iou = 0.
            avg_loss = -1

        torch.cuda.empty_cache()
        if args.local_rank == 0:
            logging.info("Epoch: {} train miou: {:.2f}".format(epoch+1, 100*train_mIoU))
            if temp_iou > best_valid_iou:
                best_valid_iou = temp_iou
                best_epoch = epoch

                if model_ema is not None:
                    torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(args.save, 'best_checkpoint.pth.tar'))
                else:
                    torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(args.save, 'best_checkpoint.pth.tar'))

            logger.add_scalar("mIoU/val", temp_iou, epoch)
            logging.info("[Epoch %d/%d] valid mIoU %.4f eval loss %.4f"%(epoch + 1, args.epochs, temp_iou, avg_loss))
            logging.info("Best valid mIoU %.4f Epoch %d"%(best_valid_iou, best_epoch))

            if model_ema is not None:
                torch.save({
                    'start_epoch': epoch + 1,
                    'state_dict': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                }, os.path.join(args.save, 'last.pth.tar'))
            else:
                torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(args.save, 'last.pth.tar'))


def train(train_loader, det2_dataset, model, model_ema, criterion, num_classes, lr_scheduler, optimizer, logger, epoch, args, cfg):

    model.train()
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_std = cfg.MODEL.PIXEL_STD
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()

    metric = seg_metrics.Seg_Metrics(n_classes=num_classes)
    lamb = 0.2
    # for i, sample in enumerate(train_loader):
    for i in range(len(train_loader)):
        cur_iter = epoch * len(train_loader) + i
        lr_scheduler(optimizer, cur_iter)
        # inputs = sample['image'].cuda(non_blocking=True)
        # target = sample['semantic'].cuda(non_blocking=True)

        det2_data = next(det2_dataset)
        det2_inputs = [x["image"].cuda(non_blocking=True) for x in det2_data]
        det2_inputs = [(x - pixel_mean) / pixel_std for x in det2_inputs]
        det2_inputs = ImageList.from_tensors(det2_inputs, args.size_divisibility).tensor

        det2_targets = [x["sem_seg"].cuda(non_blocking=True) for x in det2_data]
        det2_targets = ImageList.from_tensors(det2_targets, args.size_divisibility, args.ignore).tensor

        N = det2_inputs.size(0)

        loss = 0
        description = ""

        logits8, logits16, logits32 = model(det2_inputs)
        loss = loss + criterion(logits8, det2_targets)
        if logits16 is not None:
            loss = loss + lamb * criterion(logits16, det2_targets)
        if logits32 is not None:
            loss = loss + lamb * criterion(logits32, det2_targets)

        inter, union = seg_metrics.batch_intersection_union(logits8.data, det2_targets, num_classes)
        inter = reduce_tensor(torch.FloatTensor(inter).cuda(), args.world_size)
        union = reduce_tensor(torch.FloatTensor(union).cuda(), args.world_size)
        metric.update(inter.cpu().numpy(), union.cpu().numpy(), N)

        if args.local_rank == 0:
            description += "[mIoU%d: %.3f]"%(0, metric.get_scores())

        torch.cuda.synchronize()

        reduced_loss = loss
        reduced_loss = reduce_tensor(reduced_loss.data, args.world_size)
        if args.local_rank == 0 and i % 20 == 0:
            logger.add_scalar('loss/train', reduced_loss, epoch*len(train_loader)+i)
            logging.info('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {4:.4f}'.format(
                epoch + 1, i + 1, len(train_loader), lr_scheduler.get_lr(optimizer), reduced_loss))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

    return metric.get_scores()


def validation(val_loader, model, criterion, n_classes, args, cal_miou=True):
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.eval()
    test_loss = 0.0

    hist_size = (n_classes, n_classes)
    hist = torch.zeros(hist_size, dtype=torch.float32).cuda()

    for i, sample in enumerate(val_loader):
        sample = to_cuda(sample, device)
        image = sample['image']
        target = sample['semantic']

        N, H, W = target.shape
        probs = torch.zeros((N, n_classes, H, W)).cuda()
        probs.requires_grad = False

        torch.cuda.synchronize()
        if args.local_rank==0:
            logging.info("Evaluation [{}/{}]".format(i+1, len(val_loader)))
        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output, 1)
            probs += prob
            loss = criterion(output, target).detach().data
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            test_loss += loss

            if args.eval_flip:
                output = model(torch.flip(image, dims=(3,)))
                output = torch.flip(output, dims=(3,))
                prob = F.softmax(output, 1)
                probs += prob
                loss = criterion(output, target).detach().data
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                test_loss += loss

        if cal_miou:
            # probs = probs.data.numpy()
            preds = torch.argmax(probs, dim=1)
            hist_once = compute_hist(preds, target, n_classes, args.ignore)
            hist = hist + hist_once
        
        torch.cuda.synchronize()


    if args.eval_flip:
        avg_loss = test_loss / 2*len(val_loader)
    else:
        avg_loss = test_loss / len(val_loader)

    if cal_miou:
        # hist = torch.tensor(hist).cuda()
        dist.all_reduce(hist, dist.ReduceOp.SUM)
        hist = hist.cpu().numpy().astype(np.float32)
        IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.mean(IOUs)
    else:
        mIOU = -avg_loss

    return mIOU*100, avg_loss

if __name__ == '__main__':
    main()
