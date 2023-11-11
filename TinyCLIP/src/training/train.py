import json
import logging
import math
import os
import psutil
import functools
import time
from collections import defaultdict

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from collections import UserDict

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss
from open_clip.clip_soft_loss import ClipSoftLoss
from timm.utils.model import unwrap_model
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from training.optimizer import build_optimizer
from training.scheduler import cosine_lr, cosine_lr_start, step_lr, cosine_lr_start_nowarmup
import torch.distributed as dist
from training.my_meter import AverageMeter, reduce_tensor


def _stack2cat(items):
    if isinstance(items, torch.Tensor):
        shape = items.shape
        shape = (shape[0] * shape[1],) + shape[2:]
        return items.view(shape)
    elif isinstance(items, (list, tuple)):
        return [_stack2cat(e) for e in items]
    elif isinstance(items, (dict, UserDict)):
        return {k: _stack2cat(v) for k, v in items.items()}
    else:
        raise TypeError(f'Unsupported type {type(items)}')


def cat_items(items):
    # items: [Tensor, Tensor, ...] -> Tensor,
    # [(Tensor, Tensor), (Tensor, Tensor)] -> (Tensor, Tensor)
    # [(Tensor, [Tensor, Tensor]), (Tensor, [Tensor, Tensor])] -> (Tensor, [Tensor, Tensor])
    items = default_collate(items)  # stack of items
    # stack -> cat
    items = _stack2cat(items)
    return items


def infer_chunks(fn, x, times):
    if times == 1:
        return fn(x)
    ys = []
    for e in x.chunk(times):
        ys.append(fn(e))
    return cat_items(ys)


def check_last_batch(it):
    '''
    input: iterator
    return: (item, is_last_batch)
    '''
    last = None
    for x in it:
        if last is not None:
            yield last, False
        last = x
    if last is not None:
        yield last, True


NAN_LOSS_CNT = 0


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, scheduler_l0, args, tb_writer=None, start_iter=0, zs=None):

    global NAN_LOSS_CNT

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    image_autocast = get_autocast(args.image_precision)
    text_autocast = get_autocast(args.text_precision)
    logit_autocast = get_autocast(args.logit_precision)

    model.set_autocast(
        image_autocast=image_autocast,
        text_autocast=text_autocast,
        logit_autocast=logit_autocast)

    teacher_autocast = torch.cuda.amp.autocast

    model_without_ddp = unwrap_model(model)

    distillation = args.distillation
    if distillation:
        teacher_model = model_without_ddp.teacher[0]

    model.train()
    loss_kwargs = dict(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    if start_iter == 0:
        # set epoch in process safe manner via sampler or shared_epoch
        data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader

    dataloader.device = args.device
    if distillation:
        soft_loss_fn = ClipSoftLoss(**loss_kwargs)  # , ignore_diag=True)
    else:
        soft_loss_fn = None

    hard_loss_fn = ClipLoss(**loss_kwargs)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None and start_iter == 0:
        # [DO NOT REMOVE IT] it will call set_epoch even if sampler is not a DistributedSampler.
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    metrics = defaultdict(AverageMeter)
    end = time.time()
    batch_size = dataloader.batch_size
    samples_per_epoch = dataloader.num_samples
    total_batch_size = batch_size * args.world_size
    num_feed_images = samples_per_epoch * epoch + start_iter * total_batch_size
    num_feed_images_after_epoch = samples_per_epoch * (epoch + 1)
    all_num_feed_images = (
        int(samples_per_epoch * args.epochs) // total_batch_size * total_batch_size)

    # for float epoch
    is_last_epoch = (epoch + 1 >= args.epochs)
    samples_per_epoch_r = samples_per_epoch if not is_last_epoch else all_num_feed_images - \
        epoch * samples_per_epoch
    num_batches_per_epoch_r = samples_per_epoch_r // total_batch_size

    eval_freq = int(os.getenv('EVAL_FREQ', 1000))
    save_freq = int(os.getenv('SAVE_FREQ', 1000))

    # define model_fn and loss_fn
    infer_teacher_image = True

    def loss_fn(student_outputs,
                teacher_outputs):
        image_features = student_outputs['image_features']
        text_features = student_outputs['text_features']
        logit_scale = student_outputs['logit_scale']

        teacher_image_features = teacher_outputs['image_features']
        teacher_text_features = teacher_outputs['text_features']
        teacher_logit_scale = teacher_outputs['logit_scale']
        labels = teacher_outputs['labels']

        losses = dict()
        if distillation:
            if args.distillation_alpha > 0.0 and args.distillation_weight > 0.0:
                soft_loss_weight = args.distillation_alpha * args.distillation_weight
                img2text_loss, text2img_loss = soft_loss_fn(image_features, text_features, logit_scale,
                                                            teacher_image_features, teacher_text_features, teacher_logit_scale,
                                                            labels=labels,
                                                            average_two_losses=False,
                                                            )

                img2text_loss *= 0.5 * soft_loss_weight
                text2img_loss *= 0.5 * soft_loss_weight
                soft_loss = img2text_loss + text2img_loss

                losses['soft_loss'] = soft_loss

                metrics['soft_img2text_loss'].update(img2text_loss.item())
                metrics['soft_text2img_loss'].update(text2img_loss.item())

            # Hard Loss
            if args.distillation_alpha < 1.0 and args.distillation_weight > 0.0:
                hard_loss = hard_loss_fn(image_features, text_features, logit_scale) *\
                    ((1.0 - args.distillation_alpha) * args.distillation_weight)
                losses['hard_loss'] = hard_loss
        else:
            losses['loss'] = hard_loss_fn(
                image_features, text_features, logit_scale)

        total_loss = 0
        for k, v in losses.items():
            metrics[k].update(v.item())
            assert v.requires_grad, k
            total_loss += v
        return total_loss

    def grad_cache_loss_fn(student_outputs, teacher_outputs):
        image_features, text_features, logit_scale = student_outputs
        student_outputs = dict(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )
        return loss_fn(student_outputs, teacher_outputs)

    gpu_mem_info = torch.cuda.mem_get_info()
    gpu_memory_used = (gpu_mem_info[1] - gpu_mem_info[0]) / (1024 ** 3)
    metrics['gpu_memory'].update(gpu_memory_used)

    cpu_mem_info = psutil.virtual_memory()
    cpu_memory_used = cpu_mem_info.used / (1024 ** 3)
    metrics['cpu_memory'].update(cpu_memory_used)

    rest_shm = psutil.disk_usage('/dev/shm').free / (1024 ** 3)
    metrics['rest_shm'].update(rest_shm)

    def forward_backward_fn(model, images, texts, outputs_no_grad):
        image_feat_no_grad, text_feat_no_grad, logit_scale_no_grad = outputs_no_grad
        if args.lock_image:
            images = None
        if args.lock_text:
            texts = None

        with autocast():
            image_feat, text_feat, logit_scale = model(
                images, texts, normalized=True)

        if image_feat is None:
            image_feat = image_feat_no_grad
        if text_feat is None:
            text_feat = text_feat_no_grad
        return image_feat, text_feat, logit_scale

    def naive_model_fn(student_inputs, teacher_outputs, total_loss_flag=True):
        images, texts = student_inputs
        with autocast():

            # clean outputs first to avoid the error when using MXS
            outputs_no_grad = [None, None, None]
            student_outputs = forward_backward_fn(
                model, images, texts, outputs_no_grad)
            del images, texts, student_inputs

            loss = grad_cache_loss_fn(student_outputs, teacher_outputs)

            use_image_mask = getattr(
                model.image_encoder_without_ddp, 'l0_module', None) is not None
            use_text_mask = getattr(
                model.text_encoder_without_ddp, 'l0_module', None) is not None
            if total_loss_flag and use_image_mask and use_text_mask:
                img_mask = model.image_encoder_without_ddp.l0_module
                txt_mask = model.text_encoder_without_ddp.l0_module
                all_para_txt = txt_mask.prunable_model_size
                all_para_img = img_mask.prunable_model_size
                remain_para_txt = txt_mask.get_num_parameters_and_constraint(
                    "hidden" in txt_mask.types)
                remain_para_img = img_mask.get_num_parameters_and_constraint(
                    "hidden" in img_mask.types)
                expected_sparsity = 1 - \
                    (remain_para_txt + remain_para_img) / \
                    (all_para_txt + all_para_img)
                target_sparsity_img = img_mask.get_target_sparsity(
                    step) if img_mask.lagrangian_warmup > 0 else img_mask.target_sparsity
                target_sparsity_txt = txt_mask.get_target_sparsity(
                    step) if txt_mask.lagrangian_warmup > 0 else txt_mask.target_sparsity
                target_sparsity = (target_sparsity_img +
                                   target_sparsity_txt) / 2
                lambda_1_ = (img_mask.lambda_1 + txt_mask.lambda_1) / 2
                lambda_2_ = (img_mask.lambda_2 + txt_mask.lambda_2) / 2
                zero = torch.tensor(0.0, device=expected_sparsity.device)
                total_lagrangian_loss = (
                    lambda_1_ * torch.maximum(target_sparsity - expected_sparsity, zero) +
                    lambda_2_ *
                    torch.maximum(target_sparsity -
                                  expected_sparsity, zero).square()
                )
                loss = loss + total_lagrangian_loss
                metrics['all_expected_sparsity'].update(expected_sparsity)
                metrics['vision_expected_sparsity'].update(
                    1 - remain_para_img / all_para_img)
                metrics['text_expected_sparsity'].update(
                    1 - remain_para_txt / all_para_txt)
                metrics['all_target_sparsity'].update(target_sparsity)
                metrics['all_lagran_loss'].update(total_lagrangian_loss)
            else:
                if use_image_mask:
                    lagran_loss, expected_sparsity, target_sparsity = \
                        model.image_encoder_without_ddp.l0_module.lagrangian_regularization(
                            step)
                    loss = loss + lagran_loss
                    metrics['vision_expected_sparsity'].update(
                        expected_sparsity)
                    metrics['vision_target_sparsity'].update(target_sparsity)
                    metrics['vision_lagran_loss'].update(lagran_loss)
                if use_text_mask:
                    lagran_loss, expected_sparsity, target_sparsity = \
                        model.text_encoder_without_ddp.l0_module.lagrangian_regularization(
                            step)
                    loss = loss + lagran_loss
                    metrics['text_expected_sparsity'].update(expected_sparsity)
                    metrics['text_target_sparsity'].update(target_sparsity)
                    metrics['text_lagran_loss'].update(lagran_loss)

            scaler.scale(loss).backward()
            return loss

    grad_cache = naive_model_fn

    def teacher_image_fn(images):
        feat = teacher_model.encode_image(images)
        outputs = torch.tensor([])
        return F.normalize(feat, dim=-1), outputs

    def teacher_text_fn(texts):
        feat = teacher_model.encode_text(texts)
        outputs = torch.tensor([])
        return F.normalize(feat, dim=-1), outputs

    for (i, batch), is_last_batch in check_last_batch(enumerate(dataloader, start=start_iter)):
        step = num_batches_per_epoch * epoch + i
        num_feed_images += total_batch_size

        if step == args.prune_step and model.image_encoder_without_ddp.l0_module is not None and model.text_encoder_without_ddp.l0_module is not None:
            logging.info('=== FUSE MASK IMAGE ===')
            num_params_before_fuse = sum(
                p.numel() for p in model.image_encoder_without_ddp.parameters() if p.requires_grad)
            with torch.no_grad():
                model.image_encoder_without_ddp.eval()
                image = torch.randn((1, 3, 224, 224), device='cuda')
                model.image_encoder_without_ddp(image)
                model.image_encoder_without_ddp = model.image_encoder_without_ddp.prune()
            assert hasattr(model.image_encoder_without_ddp, 'l0_module')
            model.image_encoder_without_ddp.l0_module = None
            num_params_after_fuse = sum(
                p.numel() for p in model.image_encoder_without_ddp.parameters() if p.requires_grad)
            logging.info(
                f'=> fuse MASK image: {num_params_before_fuse} -> {num_params_after_fuse}')

            logging.info('=== FUSE MASK TEXT ===')
            num_params_before_fuse = sum(
                p.numel() for p in model.text_encoder_without_ddp.parameters() if p.requires_grad)
            with torch.no_grad():
                model.text_encoder_without_ddp.eval()
                text = torch.randint(0, 100, (1, 77), device='cuda')
                model.text_encoder_without_ddp(text)
                model.text_encoder_without_ddp = model.text_encoder_without_ddp.prune()
            assert hasattr(model.text_encoder_without_ddp, 'l0_module')
            model.text_encoder_without_ddp.l0_module = None
            num_params_after_fuse = sum(
                p.numel() for p in model.text_encoder_without_ddp.parameters() if p.requires_grad)
            logging.info(
                f'=> fuse MASK text: {num_params_before_fuse} -> {num_params_after_fuse}')

            # results = evaluate(model, data, epoch, args)
            if args.distributed and not args.horovod:
                if args.use_bn_sync:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                        model)
                ddp_args = {}
                if args.ddp_static_graph:
                    # this doesn't exist in older PyTorch, arg only added if enabled
                    ddp_args['static_graph'] = True
                ddp_fn = functools.partial(
                    torch.nn.parallel.DistributedDataParallel, device_ids=[device], **ddp_args)
                model.ddpify(ddp_fn)
                model_without_ddp = model

            args.prune_image = False
            args.prune_text = False
            use_mask = False

            optimizer = build_optimizer(args, model)
            scheduler = cosine_lr_start_nowarmup(
                optimizer[0:3], args.lr, num_batches_per_epoch * args.epochs, args.prune_step)

        scheduler(step)
        if scheduler_l0 != None:
            scheduler_l0(step)

        if len(batch) == 2:
            images, texts = batch
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            labels = None
        else:
            images, texts, labels = batch
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        metrics['data_time'].update(time.time() - end)
        for opt in optimizer:
            opt.zero_grad()

        if distillation:
            # infer teacher

            if args.logit_scale is not None:
                teacher_model.logit_scale.fill_(math.log(args.logit_scale))

            with teacher_autocast():
                with torch.no_grad():
                    if infer_teacher_image:
                        teacher_image_features, teacher_image_outputs = infer_chunks(
                            teacher_image_fn, images, 1)
                    else:
                        teacher_image_features = teacher_image_outputs = None
                    teacher_text_features, teacher_text_outputs = infer_chunks(
                        teacher_text_fn, texts, 1)
                    teacher_logit_scale = teacher_model.logit_scale.exp()

        else:
            teacher_image_features = teacher_image_outputs = None
            teacher_text_features = teacher_text_outputs = None
            teacher_logit_scale = None

        grad_norm = None
        # detach and it has been backwarded
        infer_student_image = not args.use_teacher_image
        infer_student_text = not args.use_teacher_text

        student_inputs = []
        for x, used in zip([images, texts], [infer_student_image, infer_student_text]):
            if used:
                student_inputs.append(x)
            else:
                student_inputs.append(None)

        use_mask = args.prune_image or args.prune_text
        used_optimizer = []
        for opt, used in zip(optimizer, [
            infer_student_image and not args.lock_image,
            infer_student_text and not args.lock_text,
            True,
            use_mask
        ]):
            if used:
                used_optimizer.append(opt)

        # append optimizer

        teacher_outputs = dict(
            image_features=teacher_image_features,
            text_features=teacher_text_features,
            logit_scale=teacher_logit_scale,
            image_outputs=teacher_image_outputs,
            text_outputs=teacher_text_outputs,
            labels=labels,
        )

        total_loss = grad_cache(
            student_inputs, teacher_outputs=teacher_outputs, total_loss_flag=args.total_loss_flag)
        skip_this_step = False

        # check nan loss
        if not torch.isfinite(total_loss):
            NAN_LOSS_CNT += 1
            if NAN_LOSS_CNT > 100:
                print(
                    f'WARNING: non-finite loss, ending training loss: {total_loss}')
                return 'non-finite loss'
            skip_this_step = True
            print(
                f'WARNING: non-finite loss, skip this step. loss: {total_loss}, nan_loss_cnt: {NAN_LOSS_CNT}')
        else:
            NAN_LOSS_CNT = 0

        '''
        a potential bug:
            there are three branches: image, text, logit
            each optimizer has its own `found_inf_per_device`.
            The three `found_inf_per_device` should be synced, otherwise a branch will be updated with wrong gradients?
        '''
        # check loss
        for opt in used_optimizer:
            scaler.unscale_(opt)

        # sync found_inf_per_device
        found_inf = sum(
            sum(v.item() for v in scaler._per_optimizer_states[id(
                opt)]['found_inf_per_device'].values())
            for opt in used_optimizer
        )
        if found_inf > 0:
            for opt in used_optimizer:
                for v in scaler._per_optimizer_states[id(opt)]['found_inf_per_device'].values():
                    v.fill_(True)

        if args.norm_gradient_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.norm_gradient_clip, norm_type=2.0)

        # evaluate(model, data, epoch, args, tb_writer, step=step, num_feed_images=num_feed_images)
        if not skip_this_step:
            for opt in used_optimizer:
                scaler.step(opt)
        scaler.update()

        if getattr(model.image_encoder_without_ddp, 'l0_module', None) is not None:
            model._image_encoder.module.l0_module.constrain_parameters()
            metrics['vision_lambda1'].update(
                model._image_encoder.module.l0_module.lambda_1.detach().item())
            metrics['vision_lambda2'].update(
                model._image_encoder.module.l0_module.lambda_2.detach().item())
        if getattr(model.text_encoder_without_ddp, 'l0_module', None) is not None:
            model._text_encoder.module.l0_module.constrain_parameters()
            metrics['text_lambda1'].update(
                model._text_encoder.module.l0_module.lambda_1.detach().item())
            metrics['text_lambda2'].update(
                model._text_encoder.module.l0_module.lambda_2.detach().item())

        loss_scale = scaler.state_dict()["scale"]
        metrics['loss_scale'].update(loss_scale)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            if args.logit_scale is not None:
                model_without_ddp.logit_scale.fill_(math.log(args.logit_scale))
            else:
                model_without_ddp.logit_scale.clamp_(0, math.log(100))

        batch_time_cost = time.time() - end
        metrics['batch_time'].update(batch_time_cost)
        end = time.time()

        if batch_time_cost > 0:
            metrics['throughput'].update(total_batch_size / batch_time_cost)

        batch_count = i + 1
        if is_master(args) and (i % 10 == 0 or is_last_batch):

            num_samples = batch_count * total_batch_size
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = model_without_ddp.logit_scale.exp().item()
            metrics_str = ''
            for k, v in metrics.items():
                metrics_str += '{}: {:.4f} ({:.4f})\t'.format(k, v.val, v.avg)
            logging.info(
                f"Train Epoch: {epoch} [{batch_count}/{num_batches_per_epoch_r}] [{num_samples:>{sample_digits}}/{samples_per_epoch_r} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"{metrics_str} "
                f"LR: {optimizer[0].param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer[0].param_groups[0]["lr"],
                "lr_l0": optimizer[-1].param_groups[0]["lr"]
            }

            for k, v in metrics.items():
                log_data[k] = v.val
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step,
                              'num_feed_images': num_feed_images}, step=step)

        if i > 2000:
            eval_freq = 500
        do_evaluate = ((i + 1) % eval_freq == 0 or is_last_batch)
        do_save_checkpoint = ((i + 1) % save_freq == 0 or is_last_batch)
        use_mask = args.prune_image or args.prune_text
        if step == 0 and use_mask:
            do_evaluate = True

        if ((i + 1) % eval_freq == 0 or is_last_batch) or step == 0:
            from training.viz import plot
            if args.prune_image:
                model.eval()
                layers = model._image_encoder.module.l0_module.num_hidden_layers
                hidden_size = model._image_encoder.module.l0_module.hidden_size
                heads = model._image_encoder.module.l0_module.num_attention_heads
                l0device = model._image_encoder.module.l0_module.z_logas[
                    model._image_encoder.module.l0_module.types[0]].device
                zs_img = model._image_encoder.module.l0_module()
                sparsity_img = model._image_encoder.module.l0_module.calculate_model_size(zs_img)[
                    'pruned_sparsity']
                if 'mha_z' not in zs_img.keys():
                    zs_img['mha_z'] = torch.ones([layers]).to(l0device)
                if 'ffn_z' not in zs_img.keys():
                    zs_img['ffn_z'] = torch.ones([layers]).to(l0device)
                if 'hidden_z' not in zs_img.keys():
                    zs_img['hidden_z'] = torch.ones([hidden_size]).to(l0device)
                if 'heads_z' not in zs_img.keys():
                    zs_img['heads_z'] = torch.ones(
                        [layers, 1, heads, 1, 1]).to(l0device)
                if 'intermediate_z' not in zs_img.keys():
                    zs_img['intermediate_z'] = torch.ones(
                        [layers, 1, 1, hidden_size * 4]).to(l0device)
                hidden_img = zs_img['hidden_z'].detach(
                ).cpu().squeeze().numpy()
                heads_img = zs_img['mha_z'].detach().cpu().squeeze().numpy(
                ).reshape(-1, 1) * zs_img['heads_z'].detach().cpu().squeeze().numpy()
                intermediates_img = zs_img['ffn_z'].detach().cpu().squeeze().numpy(
                ).reshape(-1, 1) * zs_img['intermediate_z'].detach().cpu().squeeze().numpy()
                fig_img = plot(heads_img, intermediates_img,
                               f"Sparsity_img: {sparsity_img:.2%}")
                if dist.get_rank() == 0 and args.wandb:
                    wandb.log({
                        "test/sparsity_img": sparsity_img,
                        "pruned_structure_img": fig_img
                    }, step=step)
                model.train()

            if args.prune_text:
                model.eval()
                layers = model._text_encoder.module.l0_module.num_hidden_layers
                hidden_size = model._text_encoder.module.l0_module.hidden_size
                heads = model._text_encoder.module.l0_module.num_attention_heads
                l0device = model._text_encoder.module.l0_module.z_logas[
                    model._text_encoder.module.l0_module.types[0]].device
                zs_txt = model._text_encoder.module.l0_module()
                sparsity_txt = model._text_encoder.module.l0_module.calculate_model_size(zs_txt)[
                    'pruned_sparsity']
                if 'mha_z' not in zs_txt.keys():
                    zs_txt['mha_z'] = torch.ones([layers]).to(l0device)
                if 'ffn_z' not in zs_txt.keys():
                    zs_txt['ffn_z'] = torch.ones([layers]).to(l0device)
                if 'hidden_z' not in zs_txt.keys():
                    zs_txt['hidden_z'] = torch.ones([hidden_size]).to(l0device)
                if 'heads_z' not in zs_txt.keys():
                    zs_txt['heads_z'] = torch.ones(
                        [layers, 1, heads, 1, 1]).to(l0device)
                if 'intermediate_z' not in zs_txt.keys():
                    zs_txt['intermediate_z'] = torch.ones(
                        [layers, 1, 1, hidden_size * 4]).to(l0device)
                hidden_txt = zs_txt['hidden_z'].detach(
                ).cpu().squeeze().numpy()
                heads_txt = zs_txt['mha_z'].detach().cpu().squeeze().numpy(
                ).reshape(-1, 1) * zs_txt['heads_z'].detach().cpu().squeeze().numpy()
                intermediates_txt = zs_txt['ffn_z'].detach().cpu().squeeze().numpy(
                ).reshape(-1, 1) * zs_txt['intermediate_z'].detach().cpu().squeeze().numpy()
                fig_txt = plot(heads_txt, intermediates_txt,
                               f"Sparsity_txt: {sparsity_txt:.2%}")
                if dist.get_rank() == 0 and args.wandb:
                    wandb.log({
                        "test/sparsity_txt": sparsity_txt,
                        "pruned_structure_txt": fig_txt
                    }, step=step)
                model.train()

        if do_evaluate:
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                evaluate(model, data, epoch, args, tb_writer,
                         step=step, num_feed_images=num_feed_images)
                model.train()

        if do_save_checkpoint and is_master(args):
            # Saving checkpoints.
            if args.save_logs:
                num_batches = len(dataloader)
                samples_per_epoch = dataloader.num_samples
                checkpoint_dict = {
                    "args": args,
                    "epoch": epoch,
                    "iter_in_epoch": i,
                    "num_batches": num_batches,
                    "samples_per_epoch": samples_per_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": [opt.state_dict() for opt in optimizer],
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                # Model EMA
                if hasattr(model_without_ddp, '_model_ema'):
                    ema_models_state = [get_state_dict(
                        model_ema) for model_ema in model_without_ddp._model_ema]
                    checkpoint_dict['model_emas'] = ema_models_state

                checkpoint_fname = os.path.join(
                    args.checkpoint_path, f"epoch_{epoch}_iter_{i}.pt")
                torch.save(
                    checkpoint_dict,
                    checkpoint_fname,
                )
                print(f"Save checkpoint to {checkpoint_fname}")

        if num_feed_images >= all_num_feed_images:
            break

    print(
        f'Feed ALL Data: {num_feed_images}/{num_feed_images_after_epoch}/{all_num_feed_images}')
    return model, optimizer, scaler, scheduler, scheduler_l0, args
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, step=None, num_feed_images=None):
    metrics = {}
    models = [model]
    names = ['']
    assert len(names) == len(models)
    for name, model_i in zip(names, models):
        model_i.eval()
        zero_shot_metrics = zero_shot_eval(model_i, data, epoch, args)
        zero_shot_metrics = dict((name + k, v)
                                 for k, v in zero_shot_metrics.items())
        metrics.update(zero_shot_metrics)

    if not metrics:
        return metrics

    if not is_master(args):
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            log = {f"val/{name}": val, 'epoch': epoch}
            extra_kwargs = dict()
            if step is not None:
                log['step'] = step
                extra_kwargs['step'] = step
            if num_feed_images is not None:
                log['num_feed_images'] = num_feed_images
            wandb.log(log, **extra_kwargs)
    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
