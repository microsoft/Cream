from __future__ import division

import argparse
import torch
# torch.multiprocessing.set_sharing_strategy('file_system') 
# for file_descriptor, but cause shm leak while nas optimizer
import os

from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work_dir', default='/cache/tmp', help='path to save log and model')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args, unparsed = parser.parse_known_args()

    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Solve SyncBN deadlock
    os.environ["NCCL_LL_THRESHOLD"] = '0'
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    cfg.work_dir = args.work_dir
    cfg.gpus = args.gpus
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    
    train_dataset = build_dataset(cfg.data.train)
        
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(), find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)
    print(model)
    print("Model have {} paramerters.".format(sum(x.numel() for x in model.parameters()) / 1e6))
    print("Model have {} backbone.".format(sum(x.numel() for x in model.module.backbone.parameters()) / 1e6))
    print("Model have {} neck.".format(sum(x.numel() for x in model.module.neck.parameters()) / 1e6))
    print("Model have {} head.".format(sum(x.numel() for x in model.module.bbox_head.parameters()) / 1e6))
    
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
