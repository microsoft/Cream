import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from datasets.data_utils import SubsetDistributedSampler
from datasets.data_utils import ImageNetPolicy

def get_search_datasets(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        config.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_data = dset.ImageFolder(
        config.test_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    num_train = len(train_data)
    indices = list(range(num_train))
    split_mid = int(np.floor(0.5 * num_train))

    train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
    valid_sampler = SubsetDistributedSampler(train_data, indices[split_mid:num_train])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]

def get_augment_datasets(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if config.use_aa:
        train_data = dset.ImageFolder(
            config.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_data = dset.ImageFolder(
            config.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
            ]))
    
    test_data = dset.ImageFolder(
        config.test_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]

