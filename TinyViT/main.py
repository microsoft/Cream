# --------------------------------------------------------
# TinyViT Main (train/validate)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Add distillation with saved teacher logits
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

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    add_common_args,\
    get_git_info

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

try:
    import wandb
except ImportError:
    wandb = None
NORM_ITER_LEN = 100


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(
        data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)

    if config.DISTILL.ENABLED:
        # we disable MIXUP and CUTMIX when knowledge distillation
        assert len(
            config.DISTILL.TEACHER_LOGITS_PATH) > 0, "Please fill in DISTILL.TEACHER_LOGITS_PATH"
        criterion = SoftTargetCrossEntropy()
    else:
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # set_epoch for dataset_train when distillation
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        if config.DISTILL.ENABLED:
            train_one_epoch_distill_using_saved_logits(
                args, config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        else:
            train_one_epoch(args, config, model, criterion,
                            data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        if is_main_process() and args.use_wandb:
            wandb.log({
                f"val/acc@1": acc1,
                f"val/acc@5": acc5,
                f"val/loss": loss,
                "epoch": epoch,
            })
            wandb.run.summary['epoch'] = epoch
            wandb.run.summary['best_acc@1'] = max_accuracy

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_one_epoch(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_distill_using_saved_logits(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    num_classes = config.MODEL.NUM_CLASSES
    topk = config.DISTILL.LOGITS_TOPK

    for idx, ((samples, targets), (logits_index, logits_value, seeds)) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets
        meters['data_time'].update(time.time() - data_tic)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        # recover teacher logits
        logits_index = logits_index.long()
        logits_value = logits_value.float()
        logits_index = logits_index.cuda(non_blocking=True)
        logits_value = logits_value.cuda(non_blocking=True)
        minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                       ) / (num_classes - topk)
        minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
        outputs_teacher = minor_value.scatter_(-1, logits_index, logits_value)

        loss = criterion(outputs, outputs_teacher)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        # compute accuracy
        real_batch_size = len(original_targets)
        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        meters['train_acc1'].update(acc1.item(), real_batch_size)
        meters['train_acc5'].update(acc5.item(), real_batch_size)
        teacher_acc1, teacher_acc5 = accuracy(
            outputs_teacher, original_targets, topk=(1, 5))
        meters['teacher_acc1'].update(teacher_acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(teacher_acc5.item(), real_batch_size)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), real_batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                acc1_meter, acc5_meter = meters['train_acc1'], meters['train_acc5']
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)
    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(args, config, data_loader, model, num_classes=1000):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if not args.only_cpu:
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
    logger.info(
        f' The number of validation samples is {int(acc1_meter.count)}')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    # we follow the throughput measurement of LeViT repo (https://github.com/facebookresearch/LeViT/blob/main/speed_test.py)
    model.eval()

    T0, T1 = 10, 60
    images, _ = next(iter(data_loader))
    batch_size, _, H, W = images.shape
    inputs = torch.randn(batch_size, 3, H, W).cuda(non_blocking=True)

    # trace model to avoid python overhead
    model = torch.jit.trace(model, inputs)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    throughput = batch_size / timing.mean().item()
    logger.info(f"batch_size {batch_size} throughput {throughput}")


if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    if config.DISTILL.TEACHER_LOGITS_PATH:
        config.DISTILL.ENABLED = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = 'nccl'

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            wandb.init(project="TinyViT", config=config_dict,
                       dir=wandb_output_path)

    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config)
