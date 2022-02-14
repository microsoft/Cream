import torch
import torch.nn as nn
from lib.utils import utils
from lib.models.loss import Loss_interactive

def search(train_loader, valid_loader, model, optimizer, w_optim, alpha_optim, layer_idx, epoch, writer, logger, config):
    # interactive retrain and kl

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses_interactive = utils.AverageMeter()
    losses_cls = utils.AverageMeter()
    losses_reg = utils.AverageMeter()

    step_num = len(train_loader)
    step_num = int(step_num * config.sample_ratio)

    cur_step = epoch*step_num
    cur_lr_search = w_optim.param_groups[0]['lr']
    cur_lr_main = optimizer.param_groups[0]['lr']
    if config.local_rank == 0:  
        logger.info("Train Epoch {} Search LR {}".format(epoch, cur_lr_search))
        logger.info("Train Epoch {} Main LR {}".format(epoch, cur_lr_main))
        writer.add_scalar('retrain/lr', cur_lr_search, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        if step > step_num:
            break

        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        #use valid data
        alpha_optim.zero_grad()
        optimizer.zero_grad()

        logits_search, emsemble_logits_search = model(val_X, layer_idx, super_flag=True)
        logits_main, emsemble_logits_main= model(val_X, layer_idx, super_flag=False)

        loss_cls = (criterion(logits_search, val_y) + criterion(logits_main, val_y)) / config.loss_alpha
        loss_interactive = Loss_interactive(emsemble_logits_search, emsemble_logits_main, config.loss_T, config.interactive_type) * config.loss_alpha

        loss_regular = 0 * loss_cls 
        if config.regular:
            reg_decay = max(config.regular_coeff * (1 - float(epoch-config.pretrain_epochs)/((config.search_iter-config.pretrain_epochs)*config.search_iter_epochs*config.regular_ratio)), 0)
            # normal cell
            op_opt = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']
            op_groups = []
            for idx in range(layer_idx, 3):
                for op_dx in op_opt:
                    op_groups.append((idx - layer_idx, op_dx))
            loss_regular = loss_regular + model.module.add_alpha_regularization(op_groups, weight_decay=reg_decay, method='L1', reduce=False)

            # reduction cell
            # op_opt = []
            op_opt = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']
            op_groups = []
            for i in range(layer_idx, 3):
                for op_dx in op_opt:
                    op_groups.append((i - layer_idx, op_dx))
            loss_regular = loss_regular + model.module.add_alpha_regularization(op_groups, weight_decay=reg_decay, method='L1', normal=False)
                
 
        loss = loss_cls + loss_interactive + loss_regular
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), config.w_grad_clip)
        optimizer.step()
        alpha_optim.step()
                    
        prec1, prec5 = utils.accuracy(logits_main, val_y, topk=(1, 5))
        if config.distributed:
            reduced_loss = utils.reduce_tensor(loss.data, config.world_size)
            reduced_loss_interactive = utils.reduce_tensor(loss_interactive.data, config.world_size)
            reduced_loss_cls = utils.reduce_tensor(loss_cls.data, config.world_size)
            reduced_loss_reg = utils.reduce_tensor(loss_regular.data, config.world_size)
            prec1 = utils.reduce_tensor(prec1, config.world_size)
            prec5 = utils.reduce_tensor(prec5, config.world_size)

        else:
            reduced_loss = loss.data
            reduced_loss_interactive = loss_interactive.data
            reduced_loss_cls = loss_cls.data
            reduced_loss_reg = loss_regular.data

        losses.update(reduced_loss.item(), N)
        losses_interactive.update(reduced_loss_interactive.item(), N)
        losses_cls.update(reduced_loss_cls.item(), N)
        losses_reg.update(reduced_loss_reg.item(), N)

        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        torch.cuda.synchronize()
        if config.local_rank == 0 and (step % config.print_freq == 0 or step == step_num):
            logger.info(
                "Train_2: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Loss_interactive {losses_interactive.avg:.3f} Losses_cls {losses_cls.avg:.3f} Losses_reg {losses_reg.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    layer_idx+1, config.layer_num, epoch+1, config.search_iter*config.search_iter_epochs, step,
                    step_num, losses=losses, losses_interactive=losses_interactive, losses_cls=losses_cls,
                    losses_reg=losses_reg, top1=top1, top5=top5))

        if config.local_rank == 0:  
            writer.add_scalar('retrain/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('retrain/top1', prec1.item(), cur_step)
            writer.add_scalar('retrain/top5', prec5.item(), cur_step)
            cur_step += 1


        w_optim.zero_grad()
        logits_search_train, _ = model(trn_X, layer_idx, super_flag=True)
        loss_cls_train = criterion(logits_search_train, trn_y)
        loss_train = loss_cls_train
        loss_train.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.parameters(), config.w_grad_clip)
        # only update w
        w_optim.step()

        # alpha_optim.step()
        if config.distributed:
            reduced_loss_cls_train = utils.reduce_tensor(loss_cls_train.data, config.world_size)
            reduced_loss_train = utils.reduce_tensor(loss_train.data, config.world_size)
        else:
            reduced_loss_cls_train = reduced_loss_cls_train.data
            reduced_loss_train = reduced_loss_train.data

        if config.local_rank == 0 and (step % config.print_freq == 0 or step == step_num-1):
            logger.info(
                "Train_1: Loss_cls: {:.3f} Loss: {:.3f}".format(
                    reduced_loss_cls_train.item(), reduced_loss_train.item())
            )


    if config.local_rank == 0:  
        logger.info("Train_2: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, config.layer_num, epoch+1, config.search_iter*config.search_iter_epochs, top1.avg))


def retrain_warmup(valid_loader, model, optimizer, layer_idx, epoch, writer, logger, super_flag, retrain_epochs, config):

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    step_num = len(valid_loader)
    step_num = int(step_num * config.sample_ratio)

    cur_step = epoch*step_num
    cur_lr = optimizer.param_groups[0]['lr']
    if config.local_rank == 0:  
        logger.info("Warmup Epoch {} LR {:.3f}".format(epoch+1, cur_lr))
        writer.add_scalar('warmup/lr', cur_lr, cur_step)

    model.train()

    for step, (val_X, val_y) in enumerate(valid_loader):
        if step > step_num:
            break

        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = val_X.size(0)

        optimizer.zero_grad()
        logits_main, _ = model(val_X, layer_idx, super_flag=super_flag)
        loss = criterion(logits_main, val_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.module.parameters(), config.w_grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits_main, val_y, topk=(1, 5))
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
                "Warmup: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f}  "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    layer_idx+1, config.layer_num, epoch+1, retrain_epochs, step,
                    step_num, losses=losses, top1=top1, top5=top5))

        if config.local_rank == 0:  
            writer.add_scalar('retrain/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('retrain/top1', prec1.item(), cur_step)
            writer.add_scalar('retrain/top5', prec5.item(), cur_step)
            cur_step += 1

    if config.local_rank == 0:  
        logger.info("Warmup: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, config.layer_num, epoch+1, retrain_epochs, top1.avg))

def validate(valid_loader, model, layer_idx, epoch, cur_step, writer, logger, super_flag, config):
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

            logits, _ = model(X, layer_idx, super_flag=False)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

            reduced_loss = loss.data

            losses.update(reduced_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            
            torch.cuda.synchronize()
            step_num = len(valid_loader)

            if (step % config.print_freq == 0 or step == step_num-1) and config.local_rank == 0:
                logger.info(
                    "Valid: Layer {}/{} Epoch {:2d}/{} Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        layer_idx+1, config.layer_num, epoch+1, config.search_iter*config.search_iter_epochs, step, step_num,
                        losses=losses, top1=top1, top5=top5))

    if config.local_rank == 0:
        writer.add_scalar('val/loss', losses.avg, cur_step)
        writer.add_scalar('val/top1', top1.avg, cur_step)
        writer.add_scalar('val/top5', top5.avg, cur_step)

        logger.info("Valid: Layer {}/{} Epoch {:2d}/{} Final Prec@1 {:.4%}".format(
            layer_idx+1, config.layer_num, epoch+1, config.search_iter*config.search_iter_epochs, top1.avg))

    return top1.avg