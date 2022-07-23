# --------------------------------------------------------
# TinyViT Data Sampler
# Copyright (c) 2022 Microsoft
# Refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
# --------------------------------------------------------
import torch
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class MyDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        padding: (bool, optional): Whether to pad the dataset. Default: ``True``.
        pair: (bool, optional): Pair output for Mixup. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 padding: bool = True,
                 pair: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.pair = pair
        self.padding = padding
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        T = self.num_replicas if not self.pair else self.num_replicas * 2
        self.total_size = len(self.dataset)
        if self.padding:
            num_parts = self.total_size // T
            has_rest = bool(self.total_size % T)
            if self.drop_last:
                self.total_size = num_parts * T
            else:
                self.total_size = (num_parts + has_rest) * T
        self.num_samples = (
            self.total_size + self.num_replicas - 1) // self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            if self.padding:
                padding_size = self.total_size - len(indices)
                # pad to total_size
                if padding_size <= len(indices):
                    indices = torch.cat(
                        [indices, indices[:padding_size]], dim=0)
                else:
                    repeat_times = (self.total_size +
                                    len(indices) - 1) // len(indices)
                    indices = indices.repeat(repeat_times)[:self.total_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        if self.pair:
            indices = indices.view(-1, 2)
        indices = indices[self.rank:self.total_size:self.num_replicas].flatten(
        ).tolist()
        assert len(indices) == self.num_samples or (
            not self.padding and len(indices) == self.num_samples - 1)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
