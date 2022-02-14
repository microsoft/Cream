from functools import partial
import numpy as np
import torch.utils.data.sampler as _sampler
from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import GroupSampler, DistributedGroupSampler, DistributedSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader


def build_dataloader_arch(dataset,
                          imgs_per_gpu,
                          workers_per_gpu,
                          num_gpus=1,
                          dist=True,
                          split_ratio=0.5,
                          **kwargs):
    
    shuffle = kwargs.get('shuffle', True)

    num_train = len(dataset)
    indices = np.array(range(num_train))
    split = int(np.floor(split_ratio * num_train))

    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size, rank)
            sampler_train = DistributedGroupSampler(dataset, imgs_per_gpu, world_size, rank, split=split, mode='train')
            sampler_val = DistributedGroupSampler(dataset, imgs_per_gpu, world_size, rank, split=split, mode='val')
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler_train = GroupSampler(dataset, imgs_per_gpu, split, 'train') if shuffle else None
        sampler_val = GroupSampler(dataset, imgs_per_gpu, split, 'val') if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_train,
        #sampler=_sampler.SubsetRandomSampler(indices[split:]),
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    data_loader_val = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_val,
        #sampler=_sampler.SubsetRandomSampler(indices[:split]),
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader_train, data_loader_val