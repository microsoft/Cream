# ---------------------------------------------------------------
# TinyViT Utils
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Add `LRSchedulerWrapper` and `divide_param_groups_by_lr_scale`
# ---------------------------------------------------------------

import copy
import torch.distributed as dist


def is_main_process():
    return dist.get_rank() == 0


class LRSchedulerWrapper:
    """
    LR Scheduler Wrapper

    This class attaches the pre-hook on the `step` functions (including `step`, `step_update`, `step_frac`) of a lr scheduler.
    When `step` functions are called, the learning rates of all layers are updated.

    Usage:
    ```
        lr_scheduler = LRSchedulerWrapper(lr_scheduler, optimizer)
    ```
    """

    def __init__(self, lr_scheduler, optimizer):
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

    def step(self, epoch):
        self.lr_scheduler.step(epoch)
        self.update_lr()

    def step_update(self, it):
        self.lr_scheduler.step_update(it)
        self.update_lr()

    def step_frac(self, frac):
        if hasattr(self.lr_scheduler, 'step_frac'):
            self.lr_scheduler.step_frac(frac)
            self.update_lr()

    def update_lr(self):
        param_groups = self.optimizer.param_groups
        for group in param_groups:
            if 'lr_scale' not in group:
                continue
            params = group['params']
            # update lr scale
            lr_scale = None
            for p in params:
                if hasattr(p, 'lr_scale'):
                    if lr_scale is None:
                        lr_scale = p.lr_scale
                    else:
                        assert lr_scale == p.lr_scale, (lr_scale, p.lr_scale)
            if lr_scale != group['lr_scale']:
                if is_main_process():
                    print('=' * 30)
                    print("params:", [e.param_name for e in params])
                    print(
                        f"change lr scale: {group['lr_scale']} to {lr_scale}")
            group['lr_scale'] = lr_scale
            if lr_scale is not None:
                group['lr'] *= lr_scale

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.lr_scheduler.load_state_dict(*args, **kwargs)


def divide_param_groups_by_lr_scale(param_groups):
    """
    Divide parameters with different lr scale into different groups.

    Inputs
    ------
    param_groups: a list of dict of torch.nn.Parameter
    ```
    # example:
    param1.lr_scale = param2.lr_scale = param3.lr_scale = 0.6
    param4.lr_scale = param5.lr_scale = param6.lr_scale = 0.3
    param_groups = [{'params': [param1, param2, param4]},
                    {'params': [param3, param5, param6], 'weight_decay': 0.}]

    param_groups = divide_param_groups_by_lr_scale(param_groups)
    ```

    Outputs
    -------
    new_param_groups: a list of dict containing the key `lr_scale`
    ```
    param_groups = [
        {'params': [param1, param2], 'lr_scale': 0.6},
        {'params': [param3], 'weight_decay': 0., 'lr_scale': 0.6}
        {'params': [param4], 'lr_scale': 0.3},
        {'params': [param5, param6], 'weight_decay': 0., 'lr_scale': 0.3}
    ]
    ```
    """
    new_groups = []
    for group in param_groups:
        params = group.pop('params')

        '''
        divide parameters to different groups by lr_scale
        '''
        lr_scale_groups = dict()
        for p in params:
            lr_scale = getattr(p, 'lr_scale', 1.0)

            # create a list if not existed
            if lr_scale not in lr_scale_groups:
                lr_scale_groups[lr_scale] = list()

            # add the parameter with `lr_scale` into the specific group.
            lr_scale_groups[lr_scale].append(p)

        for lr_scale, params in lr_scale_groups.items():
            # copy other parameter information like `weight_decay`
            new_group = copy.copy(group)
            new_group['params'] = params
            new_group['lr_scale'] = lr_scale
            new_groups.append(new_group)
    return new_groups


def set_weight_decay(model):
    skip_list = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
