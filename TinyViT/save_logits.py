# --------------------------------------------------------
# TinyViT Save Teacher Logits
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Save teacher logits
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, NativeScalerWithGradNormCount, add_common_args

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT saving sparse logits script', add_help=False)
    add_common_args(parser)
    parser.add_argument('--check-saved-logits',
                        action='store_true', help='Check saved logits')
    parser.add_argument('--skip-eval',
                        action='store_true', help='Skip evaluation')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()

    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    optimizer = None
    lr_scheduler = None

    assert config.MODEL.RESUME
    loss_scaler = NativeScalerWithGradNormCount()
    load_checkpoint(config, model_without_ddp, optimizer,
                    lr_scheduler, loss_scaler, logger)
    if not args.skip_eval and not args.check_saved_logits:
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: top-1 acc: {acc1:.1f}%, top-5 acc: {acc5:.1f}%")

    if args.check_saved_logits:
        logger.info("Start checking logits")
    else:
        logger.info("Start saving logits")

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        if args.check_saved_logits:
            check_logits_one_epoch(
                config, model, data_loader_train, epoch, mixup_fn=mixup_fn)
        else:
            save_logits_one_epoch(
                config, model, data_loader_train, epoch, mixup_fn=mixup_fn)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Saving logits time {}'.format(total_time_str))


@torch.no_grad()
def save_logits_one_epoch(config, model, data_loader, epoch, mixup_fn):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    topk = config.DISTILL.LOGITS_TOPK

    logits_manager = data_loader.dataset.get_manager()

    for idx, ((samples, targets), (keys, seeds)) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        real_batch_size = len(samples)
        meters['teacher_acc1'].update(acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(acc5.item(), real_batch_size)

        # save teacher logits
        softmax_prob = torch.softmax(outputs, -1)

        torch.cuda.synchronize()

        write_tic = time.time()
        values, indices = softmax_prob.topk(
            k=topk, dim=-1, largest=True, sorted=True)

        cpu_device = torch.device('cpu')
        values = values.detach().to(device=cpu_device, dtype=torch.float16)
        indices = indices.detach().to(device=cpu_device, dtype=torch.int16)

        seeds = seeds.numpy()
        values = values.numpy()
        indices = indices.numpy()

        # check data type
        assert seeds.dtype == np.int32, seeds.dtype
        assert indices.dtype == np.int16, indices.dtype
        assert values.dtype == np.float16, values.dtype

        for key, seed, indice, value in zip(keys, seeds, indices, values):
            bstr = seed.tobytes() + indice.tobytes() + value.tobytes()
            logits_manager.write(key, bstr)
        meters['write_time'].update(time.time() - write_tic)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Save: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} save logits takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def check_logits_one_epoch(config, model, data_loader, epoch, mixup_fn):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    topk = config.DISTILL.LOGITS_TOPK

    for idx, ((samples, targets), (saved_logits_index, saved_logits_value, seeds)) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        softmax_prob = torch.softmax(outputs, -1)

        torch.cuda.synchronize()

        values, indices = softmax_prob.topk(
            k=topk, dim=-1, largest=True, sorted=True)

        meters['error'].update(
            (values - saved_logits_value.cuda()).abs().mean().item())
        meters['diff_rate'].update(torch.count_nonzero(
            (indices != saved_logits_index.cuda())).item() / indices.numel())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Check: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} check logits takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, num_classes=1000):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        if num_classes == 1000:
            output_num_classes = output.size(-1)
            if output_num_classes == 21841:
                output = remap_layer_22kto1k(output)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    config.defrost()
    assert len(
        config.DISTILL.TEACHER_LOGITS_PATH) > 0, "Please fill in the config DISTILL.TEACHER_LOGITS_PATH"
    config.DISTILL.ENABLED = True
    if not args.check_saved_logits:
        config.DISTILL.SAVE_TEACHER_LOGITS = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # The seed changes with config, rank, world_size and epoch
    seed = config.SEED + dist.get_rank() + config.TRAIN.START_EPOCH * \
        dist.get_world_size()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
