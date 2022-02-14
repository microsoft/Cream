import torch
import torch.nn as nn
from utils import utils
from datasets import data_utils
from models.loss import CrossEntropyLabelSmooth

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
        logits, logits_aux = model(X, layer_idx=0, super_flag=True, pretrain_flag=True)
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

            logits, _ = model(X, layer_idx=0, super_flag=True, pretrain_flag=True)
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


def sample_train(train_loader, model, optimizer, epoch, writer, logger, config):
    device = torch.device("cuda")
    if config.label_smooth > 0:
        criterion = CrossEntropyLabelSmooth(config.n_classes, config.label_smooth).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

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

        all_losses = []
        all_logits = []
        for i in range(config.sample_archs):
            ### sample new arch ###
            model.module.init_arch_params(layer_idx=0)
            genotypes = []
            for i in range(config.layer_num):
                genotype, connect = model.module.generate_genotype(i)
                genotypes.append(genotype)

                model.module.genotypes[i] = genotype
                model.module.connects[i] = connect

            logits, logits_aux = model(X, layer_idx=0, super_flag=True, pretrain_flag=True, is_slim=True)
            all_logits.append(logits)
            loss = data_utils.mixup_criterion(criterion, logits, target_a, target_b, lam)
            if config.aux_weight > 0:
                # loss_aux = criterion(logits_aux, y)
                loss_aux = data_utils.mixup_criterion(criterion, logits_aux, target_a, target_b, lam)
                loss = loss + config.aux_weight * loss_aux

            all_losses.append(loss)

            '''
            for j, genotype in enumerate(genotypes):
                if config.local_rank == 0:
                    logger.info("Random stage: {} layer: {} genotype = {}".format(i, j, genotype))
            '''

        loss = torch.sum(torch.stack(all_losses))

        if config.use_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # tricks
        for p in model.module.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
        optimizer.step()

        for i, logits in enumerate(all_logits):
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            if config.distributed:
                reduced_loss = utils.reduce_tensor(all_losses[i].data, config.world_size)
                prec1 = utils.reduce_tensor(prec1, config.world_size)
                prec5 = utils.reduce_tensor(prec5, config.world_size)
            else:
                reduced_loss = all_losses[i].data


            torch.cuda.synchronize()
            if config.local_rank == 0 and (step % config.print_freq == 0 or step == step_num):
                logger.info(
                    "Train: Epoch {:2d}/{} Step {:03d}/{:03d} Sample idx {} Loss {:.3f} "
                    "Prec@(1,5) ({:.1%}, {:.1%})".format(
                        epoch+1, config.epochs, step, step_num, i,
                         reduced_loss.item(), prec1.item(), prec5.item()))

        if config.local_rank == 0:  
            writer.add_scalar('train/loss', reduced_loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1



def sample_validate(valid_loader, model, epoch, cur_step, writer, logger, config):

    model.eval()
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            for i in range(config.sample_archs):
                ### sample new arch ###
                model.module.init_arch_params(layer_idx=0)
                genotypes = []
                for i in range(config.layer_num):
                    genotype, connect = model.module.generate_genotype(i)
                    genotypes.append(genotype)

                    model.module.genotypes[i] = genotype
                    model.module.connects[i] = connect

                logits, _ = model(X, layer_idx=0, super_flag=True, pretrain_flag=True, is_slim=True)
                loss = criterion(logits, y)

                prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

                if config.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, config.world_size)
                    prec1 = utils.reduce_tensor(prec1, config.world_size)
                    prec5 = utils.reduce_tensor(prec5, config.world_size)
                else:
                    reduced_loss = loss.data
                
                torch.cuda.synchronize()
                step_num = len(valid_loader)

                if (step % config.print_freq == 0 or step == step_num-1) and config.local_rank == 0:
                    logger.info(
                        "Valid: Epoch {:2d}/{} Step {:03d}/{:03d} Sample_index {} Loss {:.3f} "
                        "Prec@(1,5) ({:.1%}, {:.1%})".format(
                            epoch+1, config.epochs, step, step_num, i,
                            reduced_loss.item(), prec1.item(), prec5.item()))

    if config.local_rank == 0:
        writer.add_scalar('val/loss', reduced_loss.item(), cur_step)
        writer.add_scalar('val/top1', prec1.item(), cur_step)
        writer.add_scalar('val/top5', prec5.item(), cur_step)

    return prec1.item(), prec5.item()


def test_sample(valid_loader, model, epoch, cur_step, writer, logger, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)


    model.module.init_arch_params(layer_idx=0)
    genotypes = []
    
    for i in range(config.layer_num):
        genotype, connect = model.module.generate_genotype(i)
        genotypes.append(genotype)

        model.module.genotypes[i] = genotype
        model.module.connects[i] = connect

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            # logits, _ = model(X, layer_idx=0, super_flag=True, pretrain_flag=True)
            logits, _ = model(X, layer_idx=0, super_flag=True, pretrain_flag=True, is_slim=True)
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