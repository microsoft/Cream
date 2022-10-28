# --------------------------------------------------------
# TinyViT Utils
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------

import torch
import torch.distributed as dist


def get_dist_backend():
    if not dist.is_available():
        return None
    if not dist.is_initialized():
        return None
    return dist.get_backend()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self._use_gpu = get_dist_backend() == 'nccl'
        self.reset()

    def reset(self):
        # local
        self._val = 0
        self._sum = 0
        self._count = 0
        # global
        self._history_avg = 0
        self._history_count = 0
        self._avg = None

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = None

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count + self._history_count

    @property
    def avg(self):
        if self._avg is None:
            # compute avg
            r = self._history_count / max(1, self._history_count + self._count)
            _avg = self._sum / max(1, self._count)
            self._avg = r * self._history_avg + (1.0 - r) * _avg
        return self._avg

    def sync(self):
        buf = torch.tensor([self._sum, self._count],
                           dtype=torch.float32)
        if self._use_gpu:
            buf = buf.cuda()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        _sum, _count = buf.tolist()
        _avg = _sum / max(1, _count)
        r = self._history_count / max(1, self._history_count + _count)

        self._history_avg = r * self._history_avg + (1.0 - r) * _avg
        self._history_count += _count

        self._sum = 0
        self._count = 0

        self._avg = None
