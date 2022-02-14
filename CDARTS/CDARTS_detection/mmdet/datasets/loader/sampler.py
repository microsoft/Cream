from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner.utils import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, split=1000, mode=None):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.split = int(np.floor(split / samples_per_gpu)) * samples_per_gpu
        self.mode = mode
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu
        if mode == 'train':
            self.num_samples = self.split
        elif mode == 'val':
            self.num_samples = self.num_samples - self.split

    def __iter__(self):
        #indices = []
        indices = np.array([], dtype='int64')
        size_flag = 0
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
                
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            
            if self.mode == 'train':
                if (size * self.split) % sum(self.group_sizes) != 0:
                    size_flag += 1
                split = int(size/sum(self.group_sizes)*self.split)
                if i == len(self.group_sizes) - 1 and size_flag != 0:
                    split += 1
                indice = indice[:split]
            elif self.mode == 'val':
                if (size * self.split) % sum(self.group_sizes) != 0:
                    size_flag += 1
                split = int(size/sum(self.group_sizes)*self.split)
                if i == len(self.group_sizes) - 1 and size_flag != 0:
                    split += 1
                indice = indice[split:]
        
            np.random.shuffle(indice)
            # indices.append(indice)
            indices = np.concatenate([indices, indice])
        _indices = np.array([], dtype='int64')
        for i in np.random.permutation(range(len(indices) // self.samples_per_gpu)):
            _indices = np.append(_indices, indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu])
        indices = _indices
      
        indices = torch.from_numpy(indices).long()
        
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None, split=1000, mode=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.split = int(np.floor(split / samples_per_gpu / self.num_replicas)) * samples_per_gpu
        self.mode = mode

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        if self.mode == 'train':
            self.num_samples = self.split
        elif self.mode == 'val':
            self.num_samples = self.num_samples - self.split
        self.total_size = self.num_samples * self.num_replicas
        self.split *= self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        #indices = np.array([])
        indices = []
        size_flag = 0
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice = np.concatenate([indice, indice[:extra]])

                if self.mode == 'train':
                    split = int(size/sum(self.group_sizes)*self.split)
                    if (size * self.split) % sum(self.group_sizes) != 0:
                        size_flag += 1
                    if i == len(self.group_sizes) - 1 and size_flag != 0:
                        split += 1
                    indice = indice[:split]
                elif self.mode == 'val':
                    split = int(size/sum(self.group_sizes)*self.split)
                    if (size * self.split) % sum(self.group_sizes) != 0:
                        size_flag += 1
                    if i == len(self.group_sizes) - 1 and size_flag != 0:
                        split += 1
                    indice = indice[split:]
                indice = indice[list(torch.randperm(int(len(indice)), generator=g))].tolist()
                indices += indice
        assert len(indices) == self.total_size
        
        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]
        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
