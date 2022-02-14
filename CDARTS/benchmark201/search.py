""" Search cell """
import os
import copy
import apex
import json
import torch
import time
import math
import torch.nn as nn
import numpy as np
import torch.distributed as dist

from tensorboardX import SummaryWriter
from models.cdarts_controller import CDARTSController
from utils.visualize import plot
from utils import utils
from datasets.data_utils import SubsetDistributedSampler
from core.search_function import search, retrain_warmup, validate
from nas_201_api import NASBench201API as API

from configs.config import SearchConfig
config = SearchConfig()

if 'cifar' in config.dataset:
    from datasets.cifar import get_search_datasets
elif 'imagenet' in config.dataset:
    from datasets.imagenet import get_search_datasets

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
if config.local_rank == 0:
    config.print_params(logger.info)

try:
    os.makedirs(config.plot_path)
except:
    pass

if config.use_apex:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
else:
    DDP = torch.nn.parallel.DistributedDataParallel
 
         
def main():
    logger.info("Logger is set - training start")


    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # TODO
    # api = None
    api = API('/home/hongyuan/benchmark/NAS-Bench-201-v1_0-e61699.pth')

    if config.distributed:
        config.gpu = config.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(config.gpu)
        # distributed init
        torch.distributed.init_process_group(backend='nccl', init_method=config.dist_url,
                world_size=config.world_size, rank=config.local_rank)

        config.world_size = torch.distributed.get_world_size()

        config.total_batch_size = config.world_size * config.batch_size
    else:
        config.total_batch_size = config.batch_size


    loaders, samplers = get_search_datasets(config)
    train_loader, valid_loader = loaders
    train_sampler, valid_sampler = samplers

    net_crit = nn.CrossEntropyLoss().cuda()
    controller = CDARTSController(config, net_crit, n_nodes=4, stem_multiplier=config.stem_multiplier)

    resume_state = None
    if config.resume:
        resume_state = torch.load(config.resume_path, map_location='cpu')

    if config.resume:
        controller.load_state_dict(resume_state['controller'])

    controller = controller.cuda()
    if config.sync_bn:
        if config.use_apex:
            controller = apex.parallel.convert_syncbn_model(controller)
        else:
            controller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(controller)

    if config.use_apex:
        controller = DDP(controller, delay_allreduce=True)
    else:
        controller = DDP(controller, device_ids=[config.gpu])

    # warm up model_search
    if config.ensemble_param:
        w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                    {"params": controller.module.super_layers.parameters()},
                                    {"params": controller.module.fc_super.parameters()},
                                    {"params": controller.module.distill_aux_head1.parameters()},
                                    {"params": controller.module.distill_aux_head2.parameters()},
                                    {"params": controller.module.ensemble_param}],
                                    lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
    else:
        w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                    {"params": controller.module.super_layers.parameters()},
                                    {"params": controller.module.fc_super.parameters()},
                                    {"params": controller.module.distill_aux_head1.parameters()},
                                    {"params": controller.module.distill_aux_head2.parameters()}],
                                    lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)


    # search training loop
    sta_search_iter = 0
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.search_iter * config.search_iter_epochs, eta_min=config.w_lr_min)
    lr_scheduler_retrain = nn.ModuleList()
    alpha_optim = nn.ModuleList()
    optimizer = nn.ModuleList()
    sub_epoch = 0

    for search_iter in range(sta_search_iter, config.search_iter):
        if search_iter < config.pretrain_epochs:
            if config.local_rank == 0:
                logger.info("####### Super model warmup #######")
            train_sampler.set_epoch(search_iter)
            retrain_warmup(train_loader, controller, w_optim, search_iter, writer, logger, True, config.pretrain_epochs, config)
            #lr_scheduler.step()
        else:
            # build new controller
            genotype = controller.module.genotype()
            controller.module.build_nas_model(genotype)

            controller_b = copy.deepcopy(controller.module)
            del controller
            controller = controller_b.cuda()

            # sync params from super layer pool
            controller.copy_params_from_super_layer()
        
            if config.sync_bn:
                if config.use_apex:
                    controller = apex.parallel.convert_syncbn_model(controller)
                else:
                    controller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(controller)

            if config.use_apex:
                controller = DDP(controller, delay_allreduce=True)
            else:
                controller = DDP(controller, device_ids=[config.gpu])

            # weights optimizer
            if config.ensemble_param:
                w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                            {"params": controller.module.super_layers.parameters()},
                                            {"params": controller.module.fc_super.parameters()},
                                            {"params": controller.module.distill_aux_head1.parameters()},
                                            {"params": controller.module.distill_aux_head2.parameters()},
                                            {"params": controller.module.ensemble_param}],
                                            lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
            else:
                w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                            {"params": controller.module.super_layers.parameters()},
                                            {"params": controller.module.fc_super.parameters()},
                                            {"params": controller.module.distill_aux_head1.parameters()},
                                            {"params": controller.module.distill_aux_head2.parameters()}],
                                            lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
            # arch_params optimizer
            alpha_optim = torch.optim.Adam(controller.module.arch_parameters(), config.alpha_lr, betas=(0.5, 0.999),
                                        weight_decay=config.alpha_weight_decay)

                                            
            if config.ensemble_param:
                optimizer = torch.optim.SGD([{"params": controller.module.feature_extractor.parameters()},
                                            {"params": controller.module.nas_layers.parameters()},
                                            {"params": controller.module.ensemble_param},
                                            {"params": controller.module.distill_aux_head1.parameters()},
                                            {"params": controller.module.distill_aux_head2.parameters()},
                                            {"params": controller.module.fc_nas.parameters()}],
                                            lr=config.nasnet_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
            else:
                optimizer = torch.optim.SGD([{"params": controller.module.feature_extractor.parameters()},
                                            {"params": controller.module.nas_layers.parameters()},
                                            {"params": controller.module.distill_aux_head1.parameters()},
                                            {"params": controller.module.distill_aux_head2.parameters()},
                                            {"params": controller.module.fc_nas.parameters()}],
                                            lr=config.nasnet_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)

            lr_scheduler_retrain = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config.search_iter_epochs, eta_min=config.w_lr_min)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                w_optim, config.search_iter * config.search_iter_epochs, eta_min=config.w_lr_min)
    
            # warmup model main
            if config.local_rank == 0:
                logger.info("####### Sub model warmup #######")
            for warmup_epoch in range(config.nasnet_warmup):
                valid_sampler.set_epoch(warmup_epoch)
                retrain_warmup(valid_loader, controller, optimizer, warmup_epoch, writer, logger, False, config.nasnet_warmup, config)
            

            lr_search = lr_scheduler.get_lr()[0]
            lr_main = lr_scheduler_retrain.get_lr()[0]

            search_epoch = search_iter

            # reset iterators
            train_sampler.set_epoch(search_epoch)
            valid_sampler.set_epoch(search_epoch)

            # training
            search(train_loader, valid_loader, controller, optimizer, w_optim, alpha_optim, search_epoch, writer, logger, config)
 
            # sync params to super layer pool
            controller.module.copy_params_from_nas_layer()
            
            # nasbench201
            if config.local_rank == 0:
                logger.info('{}'.format(controller.module._arch_parameters))
                result = api.query_by_arch(controller.module.genotype())
                logger.info('{:}'.format(result))
                cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                    cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = utils.distill(result)

                writer.add_scalars('nasbench201/cifar10', {'train':cifar10_train,'test':cifar10_test}, search_epoch)
                writer.add_scalars('nasbench201/cifar100', {'train':cifar100_train,'valid':cifar100_valid, 'test':cifar100_test}, search_epoch)
                writer.add_scalars('nasbench201/imagenet16', {'train':imagenet16_train,'valid':imagenet16_valid, 'test':imagenet16_test}, search_epoch)

                
            #lr_scheduler.step()
            #lr_scheduler_retrain.step()
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    sta_time = time.time()
    main()
    search_time = time.time() - sta_time
    search_hour = math.floor(search_time / 3600)
    search_min = math.floor(search_time / 60 - search_hour * 60)
    if config.local_rank==0:
        logger.info("Search time: hour: {} minute: {}".format(search_hour, search_min))
