import os
import time
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.models import create_model
from my_meter import AverageMeter

from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, parse_option

from models.swin_transformer_distill import SwinTransformerDISTILL

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    loss_batch = torch.sum(- targets_prob * student_likelihood, dim=-1)
    return loss_batch.mean()

def cal_relation_loss(student_attn_list, teacher_attn_list, Ar):
    layer_num = len(student_attn_list)
    relation_loss = 0.
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        B, N, Cs = student_att[0].shape
        _, _, Ct = teacher_att[0].shape
        for i in range(3):
            for j in range(3):
                # (B, Ar, N, Cs // Ar) @ (B, Ar, Cs // Ar, N)
                # (B, Ar) + (N, N)
                matrix_i = student_att[i].view(B, N, Ar, Cs//Ar).transpose(1, 2) / (Cs/Ar)**0.5
                matrix_j = student_att[j].view(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)
                As_ij = (matrix_i @ matrix_j) 

                matrix_i = teacher_att[i].view(B, N, Ar, Ct//Ar).transpose(1, 2) / (Ct/Ar)**0.5
                matrix_j = teacher_att[j].view(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)
                At_ij = (matrix_i @ matrix_j)
                relation_loss += soft_cross_entropy(As_ij, At_ij)
    return relation_loss/(9. * layer_num)

def cal_hidden_loss(student_hidden_list, teacher_hidden_list):
    layer_num = len(student_hidden_list)
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        hidden_loss +=  torch.nn.MSELoss()(student_hidden, teacher_hidden)
    return hidden_loss/layer_num

def cal_hidden_relation_loss(student_hidden_list, teacher_hidden_list):
    layer_num = len(student_hidden_list)
    B, N, Cs = student_hidden_list[0].shape
    _, _, Ct = teacher_hidden_list[0].shape
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        student_hidden = torch.nn.functional.normalize(student_hidden, dim=-1)
        teacher_hidden = torch.nn.functional.normalize(teacher_hidden, dim=-1)
        student_relation = student_hidden @ student_hidden.transpose(-1, -2)
        teacher_relation = teacher_hidden @ teacher_hidden.transpose(-1, -2)
        hidden_loss += torch.mean((student_relation - teacher_relation)**2) * 49 #Window size x Window size
    return hidden_loss/layer_num

def load_teacher_model(type='large'):
    if type == 'large':
        embed_dim = 192
        depths = [ 2, 2, 18, 2 ]
        num_heads = [ 6, 12, 24, 48 ]
        window_size = 7
    elif type == 'base':
        embed_dim = 128
        depths = [ 2, 2, 18, 2 ]
        num_heads = [ 4, 8, 16, 32 ]
        window_size = 7
    else:
        raise ValueError('Unsupported type: %s'%type)
    model = SwinTransformerDISTILL(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                # distillation
                                is_student=False)
    return model

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    if config.DISTILL.DO_DISTILL:
        logger.info(f"Loading teacher model:{config.MODEL.TYPE}/{config.DISTILL.TEACHER}")
        model_checkpoint_name = os.path.basename(config.DISTILL.TEACHER)
        if 'regnety_160' in model_checkpoint_name:
            model_teacher = create_model(
            'regnety_160',
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            global_pool='avg',
            )
            if config.DISTILL.TEACHER.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    config.DISTILL.TEACHER, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
            model_teacher.load_state_dict(checkpoint['model'])
            model_teacher.cuda()
            model_teacher.eval()
            del checkpoint
            torch.cuda.empty_cache()
        else:
            if 'base' in model_checkpoint_name:
                teacher_type = 'base'
            elif 'large' in model_checkpoint_name:
                teacher_type = 'large'
            else:
                teacher_type = None
            model_teacher = load_teacher_model(type=teacher_type)
            model_teacher.cuda()
            model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
            checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
            msg = model_teacher.module.load_state_dict(checkpoint['model'], strict=False)
            logger.info(msg)
            del checkpoint
            torch.cuda.empty_cache()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion_soft = soft_cross_entropy
    criterion_attn = cal_relation_loss
    criterion_hidden = cal_hidden_relation_loss if config.DISTILL.HIDDEN_RELATION else cal_hidden_loss


    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion_truth = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_truth = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_truth = torch.nn.CrossEntropyLoss()
    
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.DISTILL.RESUME_WEIGHT_ONLY = False
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if config.DISTILL.DO_DISTILL:
            train_one_epoch_distill(config, model, model_teacher, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, criterion_soft=criterion_soft, criterion_truth=criterion_truth, criterion_attn=criterion_attn, criterion_hidden=criterion_hidden)
        else:
            train_one_epoch(config, model, criterion_truth, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if epoch % config.EVAL_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
            acc1, acc5, loss = validate(config, data_loader_val, model, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch_distill(config, model, model_teacher, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, criterion_soft=None, criterion_truth=None, criterion_attn=None, criterion_hidden=None):

    layer_id_s_list = config.DISTILL.STUDENT_LAYER_LIST
    layer_id_t_list = config.DISTILL.TEACHER_LAYER_LIST

    model.train()
    optimizer.zero_grad()

    model_teacher.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_soft_meter = AverageMeter()
    loss_truth_meter = AverageMeter()
    loss_attn_meter = AverageMeter()
    loss_hidden_meter = AverageMeter()
    
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    teacher_acc1_meter = AverageMeter()
    teacher_acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        original_targets = targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.DISTILL.ATTN_LOSS and config.DISTILL.HIDDEN_LOSS:
            outputs, qkv_s, hidden_s = model(samples, layer_id_s_list, is_attn_loss=True, is_hidden_loss=True, is_hidden_org=config.DISTILL.HIDDEN_RELATION)
        elif config.DISTILL.ATTN_LOSS:
            outputs, qkv_s = model(samples, layer_id_s_list, is_attn_loss=True, is_hidden_loss=False, is_hidden_org=config.DISTILL.HIDDEN_RELATION)
        elif config.DISTILL.HIDDEN_LOSS:
            outputs, hidden_s = model(samples, layer_id_s_list, is_attn_loss=False, is_hidden_loss=True, is_hidden_org=config.DISTILL.HIDDEN_RELATION)
        else:
            outputs = model(samples)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
            if config.DISTILL.ATTN_LOSS or config.DISTILL.HIDDEN_LOSS:
                outputs_teacher, qkv_t, hidden_t = model_teacher(samples, layer_id_t_list, is_attn_loss=True, is_hidden_loss=True)
            else:
                outputs_teacher = model_teacher(samples)
            teacher_acc1, teacher_acc5 = accuracy(outputs_teacher, original_targets, topk=(1, 5))

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss_truth = config.DISTILL.ALPHA*criterion_truth(outputs, targets)
            loss_soft = (1.0 - config.DISTILL.ALPHA)*criterion_soft(outputs/config.DISTILL.TEMPERATURE, outputs_teacher/config.DISTILL.TEMPERATURE)
            if config.DISTILL.ATTN_LOSS:
                loss_attn= config.DISTILL.QKV_LOSS_WEIGHT * criterion_attn(qkv_s, qkv_t, config.DISTILL.AR)
            else:
                loss_attn = torch.zeros(loss_truth.shape)
            if config.DISTILL.HIDDEN_LOSS:
                loss_hidden = config.DISTILL.HIDDEN_LOSS_WEIGHT*criterion_hidden(hidden_s, hidden_t)
            else:
                loss_hidden = torch.zeros(loss_truth.shape)
            loss = loss_truth + loss_soft + loss_attn + loss_hidden

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss_truth = config.DISTILL.ALPHA*criterion_truth(outputs, targets)
            loss_soft = (1.0 - config.DISTILL.ALPHA)*criterion_soft(outputs/config.DISTILL.TEMPERATURE, outputs_teacher/config.DISTILL.TEMPERATURE)
            if config.DISTILL.ATTN_LOSS:
                loss_attn= config.DISTILL.QKV_LOSS_WEIGHT * criterion_attn(qkv_s, qkv_t, config.DISTILL.AR)
            else:
                loss_attn = torch.zeros(loss_truth.shape)
            if config.DISTILL.HIDDEN_LOSS:
                loss_hidden = config.DISTILL.HIDDEN_LOSS_WEIGHT*criterion_hidden(hidden_s, hidden_t)
            else:
                loss_hidden = torch.zeros(loss_truth.shape)
            loss = loss_truth + loss_soft + loss_attn + loss_hidden

            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        loss_soft_meter.update(loss_soft.item(), targets.size(0))
        loss_truth_meter.update(loss_truth.item(), targets.size(0))
        loss_attn_meter.update(loss_attn.item(), targets.size(0))
        loss_hidden_meter.update(loss_hidden.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))
        teacher_acc1_meter.update(teacher_acc1.item(), targets.size(0))
        teacher_acc5_meter.update(teacher_acc5.item(), targets.size(0))

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}\t'
                f'Teacher_Acc@1 {teacher_acc1_meter.avg:.3f} Teacher_Acc@5 {teacher_acc5_meter.avg:.3f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_soft {loss_soft_meter.val:.4f} ({loss_soft_meter.avg:.4f})\t'
                f'loss_truth {loss_truth_meter.val:.4f} ({loss_truth_meter.avg:.4f})\t'
                f'loss_attn {loss_attn_meter.val:.4f} ({loss_attn_meter.avg:.4f})\t'
                f'loss_hidden {loss_hidden_meter.val:.4f} ({loss_hidden_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        original_targets = targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, logger):

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

        output = model(images)

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

    loss_meter.sync()
    acc1_meter.sync()
    acc5_meter.sync()

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
