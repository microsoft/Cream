import torch
import torch.nn as nn
from lib.utils import utils
from lib.datasets import data_utils
from lib.models.loss import CrossEntropyLabelSmooth

def train(train_loader, model, optimizer, epoch, writer, logger, config):
    device = torch.device("cuda")
    if config.label_smooth > 0:
        criterion = CrossEntropyLabelSmooth(config.n_classes, config.label_smooth).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    step_num = len(train_loader)
    cur_step = epoch*step_num
    cur_lr = optimizer.param_groups[0]['lr']
    if config.local_rank == 0:  
        logger.info("Train Epoch {} LR {}".format(epoch, cur_lr))
        writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        X, target_a, target_b, lam = data_utils.mixup_data(X, y, config.mixup_alpha, use_cuda=True)

        optimizer.zero_grad()
        logits, logits_aux = model(X)
        # loss = criterion(logits, y)
        loss = data_utils.mixup_criterion(criterion, logits, target_a, target_b, lam)
        if config.aux_weight > 0:
            # loss_aux = criterion(logits_aux, y)
            loss_aux = data_utils.mixup_criterion(criterion, logits_aux, target_a, target_b, lam)
            loss = loss + config.aux_weight * loss_aux

        if config.use_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        if config.distributed:
            reduced_loss = utils.reduce_tensor(loss.data, config.world_size)
            prec1 = utils.reduce_tensor(prec1, config.world_size)
            prec5 = utils.reduce_tensor(prec5, config.world_size)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        torch.cuda.synchronize()
        if config.local_rank == 0 and (step % config.print_freq == 0 or step == step_num):
            logger.info(
                "Train: Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step,
                    step_num, losses=losses, top1=top1, top5=top5))

        if config.local_rank == 0:  
            writer.add_scalar('train/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1

    if config.local_rank == 0:  
        logger.info("Train: Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            epoch+1, config.epochs, top1.avg))

def validate(valid_loader, model, epoch, cur_step, writer, logger, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

            if config.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, config.world_size)
                prec1 = utils.reduce_tensor(prec1, config.world_size)
                prec5 = utils.reduce_tensor(prec5, config.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            
            torch.cuda.synchronize()
            step_num = len(valid_loader)

            if (step % config.print_freq == 0 or step == step_num-1) and config.local_rank == 0:
                logger.info(
                    "Valid: Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, step_num,
                        losses=losses, top1=top1, top5=top5))

    if config.local_rank == 0:
        writer.add_scalar('val/loss', losses.avg, cur_step)
        writer.add_scalar('val/top1', top1.avg, cur_step)
        writer.add_scalar('val/top5', top5.avg, cur_step)

        logger.info("Valid: Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            epoch+1, config.epochs, top1.avg))

    return top1.avg, top5.avg
