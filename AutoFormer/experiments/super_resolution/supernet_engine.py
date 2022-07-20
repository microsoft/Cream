import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from AutoFormer.lib import utils
import random
import AutoFormer.utils.utils_image as util
import time
from AutoFormer.data.dataset_sr import DatasetSR

from loguru import logger
logger.add(sys.stdout, level='DEBUG')

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config


def sample_configs_swinir(choices):
    config = {}
    rstb_num = random.choice(choices['rstb_num'])
    config['stl_num'] = random.choice(choices['stl_num'])
    config['embed_dim'] = [random.choice(choices['embed_dim'])] * rstb_num
    config['num_heads'] = [random.choice(choices['num_heads'])] * rstb_num

    # Only MLP_RATIO is allowed to vary per RSTB block
    config['mlp_ratio'] = [random.choice(choices['mlp_ratio']) for _ in range(rstb_num)]

    config['rstb_num'] = rstb_num

    logger.debug(f'Sampled config: {config}')
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    sampler=sample_configs_swinir):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        logger.debug(config)
        model_module.set_sample_config(config=config)
        # logger.debug(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sampler(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.debug("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.debug(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, sampler=sample_configs_swinir, scaling=4):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sampler(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    logger.debug(f"sampled model config: {config}")
    # parameters = model_module.get_sampled_params_numel(config)
    # logger.debug(f"sampled model parameters: {parameters}")


    for images, target in metric_logger.log_every(data_loader, 5, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                # print(images.shape)
                output = model(images)
                print(images.shape, output.shape, target.shape)
        else:
            output = model(images)
    
        E_img = util.tensor2uint(output)
        H_img = util.tensor2uint(target)
        print(scaling)
        current_psnr = util.calculate_psnr(E_img, H_img, border=scaling)

        batch_size = images.shape[0]
        metric_logger.meters['psnr'].update(current_psnr, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* AVG_PSNR {psnr.global_avg:.3f}'
          .format(psnr=metric_logger.psnr))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
