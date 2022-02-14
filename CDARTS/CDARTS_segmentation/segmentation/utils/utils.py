# ------------------------------------------------------------------------------
# Utility functions.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            name=key, meter=loss_meter_dict[key]
        )

    return msg


def to_cuda(batch, device):
    if type(batch) == torch.Tensor:
        batch = batch.to(device)
    elif type(batch) == dict:
        for key in batch.keys():
            batch[key] = to_cuda(batch[key], device)
    elif type(batch) == list:
        for i in range(len(batch)):
            batch[i] = to_cuda(batch[i], device)
    return batch


def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model
