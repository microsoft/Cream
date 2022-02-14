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
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader

from detectron2.config import configurable
from detectron2.data.build import _test_loader_from_config, trivial_batch_collator
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset

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
parser.add_argument('-c', '--config', default='../configs/ade/cydas.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--det2_cfg', type=str, default='configs/ADE20K/base.yaml', help='')
parser.add_argument('--save', type=str, default='../OUTPUT/train_', help='')
parser.add_argument('--exp_name', type=str, default='ade20k', help='')
parser.add_argument('--pretrain', type=str, default=None, help='resume path')
parser.add_argument('--resume', type=str, default='../OUTPUT/train/', help='resume path')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--num_classes", default=150, type=int)
parser.add_argument("--max_iteration", default=160000, type=int)
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
parser.add_argument('--size_divisibility', type=int, default=32, help='size_divisibility')
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

@configurable(from_config=_test_loader_from_config)
def build_batch_test_loader(dataset, *, mapper, sampler=None, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.
    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))
        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 4, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def main():
    args, args_text = _parse_args()
        
    # dist init
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    args.save = args.save + args.exp_name

    # detectron2 data loader ###########################
    # det2_args = default_argument_parser().parse_args()
    det2_args = args
    det2_args.config_file = args.det2_cfg
    cfg = setup(det2_args)
    mapper = DatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
    det2_dataset = iter(build_detection_train_loader(cfg, mapper=mapper))
    det2_val = build_batch_test_loader(cfg, cfg.DATASETS.TEST[0])
    len_det2_train = 20210 // cfg.SOLVER.IMS_PER_BATCH

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

    num_classes = args.num_classes

    with open(args.json_file, 'r') as f:
        # dict_a = json.loads(f, cls=NpEncoder)
        model_dict = json.loads(f.read())

    width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    model = Network(Fch=args.Fch, num_classes=num_classes, stem_head_width=(args.stem_head_width, args.stem_head_width))
    last = model_dict["lasts"]

    if args.local_rank == 0:
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (3, args.eval_height, args.eval_width), as_strings=True,
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
        max_iteration = args.epochs * len_det2_train
        lr_scheduler = Iter_LR_Scheduler(args, max_iteration, len_det2_train)

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
    temp_iou = 0.
    avg_loss = -1

    logging.info("rank: {} world_size: {}".format(args.local_rank, args.world_size))
    for epoch in range(start_epoch, args.epochs):
        if args.local_rank == 0:
            logging.info(args.load_path)
            logging.info(args.save)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        drop_prob = args.drop_path_prob * epoch / args.epochs
        # model.module.drop_path_prob(drop_prob)

        train_mIoU = train(len_det2_train, det2_dataset, model, model_ema, ohem_criterion, num_classes, lr_scheduler, optimizer, logger, epoch, args, cfg)

        # torch.cuda.empty_cache()

        # if epoch > args.epochs // 3:
        if epoch >= 0:
            temp_iou, avg_loss = validation(det2_val, eval_model, ohem_criterion, num_classes, args, cfg)

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


def train(len_det2_train, det2_dataset, model, model_ema, criterion, num_classes, lr_scheduler, optimizer, logger, epoch, args, cfg):

    model.train()
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_std = cfg.MODEL.PIXEL_STD
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()

    metric = seg_metrics.Seg_Metrics(n_classes=num_classes)
    lamb = 0.2
    # for i, sample in enumerate(train_loader):
    for i in range(len_det2_train):
        cur_iter = epoch * len_det2_train + i
        lr_scheduler(optimizer, cur_iter)

        det2_data = next(det2_dataset)
        det2_inputs = [x["image"].cuda(non_blocking=True) for x in det2_data]
        det2_inputs = [(x - pixel_mean) / pixel_std for x in det2_inputs]
        det2_inputs = ImageList.from_tensors(det2_inputs, args.size_divisibility).tensor

        b, c, h, w = det2_inputs.shape
        if h % 32 != 0 or w % 32 != 0:
            logging.info("pass bad data!")
            continue

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
            logger.add_scalar('loss/train', reduced_loss, epoch*len_det2_train+i)
            logging.info('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {4:.4f}'.format(
                epoch + 1, i + 1, len_det2_train, lr_scheduler.get_lr(optimizer), reduced_loss))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

    return metric.get_scores()


def validation(val_loader, model, criterion, n_classes, args, cfg):
    device = torch.device('cuda:{}'.format(args.local_rank))

    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_std = cfg.MODEL.PIXEL_STD
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()

    model.eval()
    test_loss = 0.0
    hist_size = (n_classes, n_classes)
    hist = torch.zeros(hist_size, dtype=torch.float32).cuda()

    for i, sample in enumerate(val_loader):
        image = [x["image"].cuda(non_blocking=True) for x in sample]
        image = [(x - pixel_mean) / pixel_std for x in image]
        image = ImageList.from_tensors(image, args.size_divisibility).tensor

        target = [x["sem_seg"].cuda(non_blocking=True) for x in sample]
        target = ImageList.from_tensors(target, args.size_divisibility, args.ignore).tensor

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


        preds = torch.argmax(probs, dim=1)
        hist_once = compute_hist(preds, target, n_classes, args.ignore)
        hist = hist + hist_once
        torch.cuda.synchronize()


    if args.eval_flip:
        avg_loss = test_loss / 2*len(val_loader)
    else:
        avg_loss = test_loss / len(val_loader)

    dist.all_reduce(hist, dist.ReduceOp.SUM)
    hist = hist.cpu().numpy().astype(np.float32)
    IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
    mIOU = np.mean(IOUs)

    return mIOU*100, avg_loss

if __name__ == '__main__':
    main()
