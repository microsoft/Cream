# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import time
import torchvision
import torch.nn.functional as F

from lib.utils.util import *
from lib.utils.train_supernet import *


def train_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        cfg,
        args,
        sta_num=None,
        est=None,
        val_loader=None,
        best_children_pool=None,
        logger=None,
        saved_val_images=None,
        saved_val_labels=None,
        lr_scheduler=None,
        saver=None,
        output_dir='',
        model_ema=None,
        CHOICE_NUM=4,
        local_rank=0):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    kd_losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        print(batch_idx)
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        # get random architectures
        prob = get_prob(cfg, best_children_pool)
        random_cand = get_cand_with_prob(CHOICE_NUM, prob, sta_num=sta_num)
        random_cand.insert(0, [0])
        random_cand.append([0])

        # evaluate FLOPs of candidates
        cand_flops = est.get_flops(random_cand)

        if epoch > cfg.SUPERNET.META_STA_EPOCH and batch_idx > 0 and batch_idx % cfg.SUPERNET.UPDATE_ITER == 0:
            slice = cfg.SUPERNET.SLICE
            x = deepcopy(input[:slice].clone().detach())

            if len(best_children_pool) > 0:
                meta_value, teacher_cand = select_teacher(
                    cfg, best_children_pool, model, random_cand)

            u_output = model(x, random_cand)
            u_teacher_output = model(x, teacher_cand)
            u_soft_label = F.softmax(u_teacher_output, dim=1)
            kd_loss = meta_value * \
                cross_entropy_loss_with_soft_target(u_output, u_soft_label)
            optimizer.zero_grad()

            grad_1 = torch.autograd.grad(
                kd_loss,
                model.module.rand_parameters(random_cand),
                create_graph=True)

            def raw_sgd(w, g):
                return g * optimizer.param_groups[-1]['lr'] + w

            students_weight = [
                raw_sgd(
                    p, grad_item) for p, grad_item in zip(
                    model.module.rand_parameters(random_cand), grad_1)]

            # update student weights
            for weight, grad_item in zip(
                    model.module.rand_parameters(random_cand), grad_1):
                weight.grad = grad_item
            torch.nn.utils.clip_grad_norm_(
                model.module.rand_parameters(random_cand), 1)
            optimizer.step()
            for weight, grad_item in zip(
                    model.module.rand_parameters(random_cand), grad_1):
                del weight.grad

            held_out_x = input[slice:slice * 2].clone()
            output_2 = model(held_out_x, random_cand)
            valid_loss = loss_fn(output_2, target[slice:slice * 2])
            optimizer.zero_grad()

            grad_student_val = torch.autograd.grad(
                valid_loss, model.module.rand_parameters(random_cand), retain_graph=True)

            grad_teacher = torch.autograd.grad(
                students_weight[0],
                model.module.rand_parameters(
                    teacher_cand,
                    cfg.SUPERNET.PICK_METHOD == 'meta'),
                grad_outputs=grad_student_val)

            # update teacher model
            for weight, grad_item in zip(model.module.rand_parameters(
                    teacher_cand, cfg.SUPERNET.PICK_METHOD == 'meta'), grad_teacher):
                weight.grad = grad_item
            torch.nn.utils.clip_grad_norm_(
                model.module.rand_parameters(
                    random_cand, cfg.SUPERNET.PICK_METHOD == 'meta'), 1)
            optimizer.step()
            for weight, grad_item in zip(model.module.rand_parameters(
                    teacher_cand, cfg.SUPERNET.PICK_METHOD == 'meta'), grad_teacher):
                del weight.grad

            for item in students_weight:
                del item
            del grad_teacher, grad_1, grad_student_val, x, held_out_x
            del valid_loss, kd_loss, u_soft_label, u_output, u_teacher_output, output_2

        # get_best_teacher
        if len(best_children_pool) > 0:
            meta_value, teacher_cand = select_teacher(
                cfg, best_children_pool, model, random_cand)

        if len(best_children_pool) == 0:
            output = model(input, random_cand)
            loss = loss_fn(output, target)
            kd_loss = loss
        elif epoch <= cfg.SUPERNET.META_STA_EPOCH:
            output = model(input, random_cand)
            loss = loss_fn(output, target)
        else:
            output = model(input, random_cand)
            with torch.no_grad():
                teacher_output = model(input, teacher_cand).detach()
                soft_label = F.softmax(teacher_output, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
            valid_loss = loss_fn(output, target)
            loss = (meta_value * kd_loss + (2 - meta_value) * valid_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        if cfg.NUM_GPU == 1:
            reduced_loss = loss.data
        else:
            reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
            prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
            prec5 = reduce_tensor(prec5, cfg.NUM_GPU)

        # best_children_pool = sorted(best_children_pool, reverse=True)
        if epoch > cfg.SUPERNET.META_STA_EPOCH and ((len(best_children_pool) < cfg.SUPERNET.POOL_SIZE) or (
                prec1 > best_children_pool[-1][1] + 5) or (prec1 > best_children_pool[-1][1] and cand_flops < best_children_pool[-1][2])):
            val_prec1 = prec1
            training_data = deepcopy(input[:cfg.SUPERNET.SLICE].detach())
            if len(best_children_pool) == 0:
                features = deepcopy(output[:cfg.SUPERNET.SLICE].detach())
            else:
                features = deepcopy(
                    teacher_output[:cfg.SUPERNET.SLICE].detach())
            best_children_pool.append(
                (val_prec1,
                 prec1,
                 cand_flops,
                 random_cand,
                 training_data,
                 F.softmax(
                     features,
                     dim=1)))
            best_children_pool = sorted(best_children_pool, reverse=True)

        if len(best_children_pool) > cfg.SUPERNET.SLICE:
            best_children_pool = sorted(best_children_pool, reverse=True)
            del best_children_pool[-1]

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        kd_losses_m.update(kd_loss.item(), input.size(0))
        prec1_m.update(prec1.item(), output.size(0))
        prec5_m.update(prec5.item(), output.size(0))

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # print(time.time() - end)
        if last_batch or batch_idx % cfg.LOG_INTERVAL == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'KD-Loss: {kd_loss.val:>9.6f} ({kd_loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        kd_loss=kd_losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * cfg.NUM_GPU / batch_time_m.val,
                        rate_avg=input.size(0) * cfg.NUM_GPU / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if cfg.SAVE_IMAGES and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            batch_idx), padding=0, normalize=True)

        if saver is not None and cfg.RECOVERY_INTERVAL and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
            saver.save_recovery(
                model,
                optimizer,
                args,
                epoch,
                model_ema=model_ema,
                batch_idx=batch_idx)

        end = time.time()

    if local_rank == 0:
        for idx, i in enumerate(best_children_pool):
            logger.info("No.{} {}".format(idx, i[:4]))

    return OrderedDict([('loss', losses_m.avg)]), best_children_pool


def validate(
        model,
        loader,
        loss_fn,
        cfg,
        log_suffix='',
        CHOICE_NUM=4,
        sta_num=None,
        local_rank=0,
        logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    # get random child architecture
    random_cand = get_cand_with_prob(CHOICE_NUM, None, sta_num=sta_num)
    random_cand.insert(0, [0])
    random_cand.append([0])

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input, random_cand)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = cfg.TTA
            if reduce_factor > 1:
                output = output.unfold(
                    0,
                    reduce_factor,
                    reduce_factor).mean(
                    dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            if cfg.NUM_GPU > 1:
                reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
                prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
                prec5 = reduce_tensor(prec5, cfg.NUM_GPU)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if local_rank == 0 and (
                    last_batch or batch_idx %
                    cfg.LOG_INTERVAL == 0):
                log_name = 'Test' + log_suffix
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m, loss=losses_m,
                        top1=prec1_m, top5=prec5_m))

    metrics = OrderedDict(
        [('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg)])

    return metrics
