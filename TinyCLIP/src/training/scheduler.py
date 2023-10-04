import numpy as np


def assign_learning_rate(optimizer, new_lr):
    if isinstance(optimizer, list):
        for opt in optimizer:
            assign_learning_rate(opt, new_lr)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr_start(optimizer, base_lr, warmup_length, steps, start_steps):
    def _lr_adjuster(step):
        if step < start_steps:
            # lr = 0.0001
            lr = 0.00005
        elif step < warmup_length + start_steps:
            lr = _warmup_lr(base_lr, warmup_length, step - start_steps)
        else:
            e = step - warmup_length - start_steps
            es = steps - warmup_length - start_steps
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr_start_nowarmup(optimizer, base_lr, steps, start_steps):
    def _lr_adjuster(step):
        if step < start_steps:
            lr = 0.0001
        else:
            e = step - start_steps
            es = steps - start_steps
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def step_lr(optimizer, start_steps):
    def _lr_adjuster(step):
        if step > start_steps:
            lr = 0
            assign_learning_rate(optimizer, lr)
            return lr
        else:
            return None
    return _lr_adjuster


def exponential_lr(optimizer, base_lr, warmup_length, steps, gamma, w):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            # lr = base_lr * gamma ** (e / es * w)
            # min_lr = base_lr * gamma ** (w)
            # w = np.log(min_lr / base_lr) / np.log(gamma)
            lr = base_lr * gamma ** (e / es * w)
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster
