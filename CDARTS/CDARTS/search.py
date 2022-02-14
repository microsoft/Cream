""" Search cell """
import _init_paths
import os
import copy
import json
import torch
import time
import math
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
from lib.models.cdarts_controller import CDARTSController
from lib.utils.visualize import plot
from lib.utils import utils
from lib.core.search_function import search, retrain_warmup

from lib.config import SearchConfig
config = SearchConfig()

if 'cifar' in config.dataset:
    from lib.datasets.cifar import get_search_datasets
elif 'imagenet' in config.dataset:
    from lib.datasets.imagenet import get_search_datasets

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
if config.local_rank == 0:
    config.print_params(logger.info)

try:
    os.makedirs(config.retrain_path)
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
    if config.param_pool_path is not None:
        param_pool = torch.load(config.param_pool_path, map_location='cpu')
        controller.load_state_dict(param_pool, strict=False)
        
    resume_state = None
    if config.resume:
        resume_state = torch.load(config.resume_path, map_location='cpu')

    sta_layer_idx = 0
    if config.resume:
        controller.load_state_dict(resume_state['controller'])
        sta_layer_idx = resume_state['sta_layer_idx']

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
    layer_idx=0
    if config.ensemble_param:
        w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                    {"params": controller.module.super_layers[layer_idx].parameters(), 'lr':config.w_lr},
                                    {"params": controller.module.super_layers[layer_idx+1:].parameters()},
                                    {"params": controller.module.fc_super.parameters()},
                                    {"params": controller.module.distill_aux_head1.parameters()},
                                    {"params": controller.module.distill_aux_head2.parameters()},
                                    {"params": controller.module.ensemble_param},
                                    {"params": controller.module.nas_layers[:layer_idx].parameters()}],
                                    lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
    else:
        w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                    {"params": controller.module.super_layers[layer_idx].parameters(), 'lr':config.w_lr},
                                    {"params": controller.module.super_layers[layer_idx+1:].parameters()},
                                    {"params": controller.module.fc_super.parameters()},
                                    {"params": controller.module.distill_aux_head1.parameters()},
                                    {"params": controller.module.distill_aux_head2.parameters()},
                                    {"params": controller.module.nas_layers[:layer_idx].parameters()}],
                                    lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)



    for layer_idx in range(sta_layer_idx, config.layer_num):
        if config.one_stage:
            if layer_idx > 0:
                break

        # clean arch params in model_search
        if config.clean_arch:
            controller.module.init_arch_params(layer_idx)

        # search training loop
        best_top1 = 0.
        best_genotypes = []
        best_connects = []
        sta_search_iter, sta_search_epoch = 0, 0
        is_best = True
        if (layer_idx == sta_layer_idx) and (resume_state is not None):
            sta_search_iter = resume_state['sta_search_iter']
            sta_search_epoch = resume_state['sta_search_epoch']
            best_top1 = resume_state['best_top1']
            best_genotypes = resume_state['best_genotypes']
            best_connects = resume_state['best_connects']
        else:
            # init model main
            if config.gumbel_sample:
                genotype, connect = controller.module.generate_genotype_gumbel(0)
            else:
                genotype, connect = controller.module.generate_genotype(0)
            for i in range(config.layer_num):
                best_genotypes.append(genotype)
                best_connects.append(connect)

        for i in range(config.layer_num):
            controller.module.genotypes[i] = best_genotypes[i]
            controller.module.connects[i] = best_connects[i]

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
                retrain_warmup(train_loader, controller, w_optim, layer_idx, search_iter, writer, logger, True, config.pretrain_epochs, config)
                #lr_scheduler.step()
            else:
                # build new controller
                for i, genotype in enumerate(best_genotypes):
                    controller.module.build_nas_layers(i, genotype, config.same_structure)

                controller_b = copy.deepcopy(controller.module)
                del controller
                controller = controller_b.cuda()
                controller.fix_pre_layers(layer_idx)

                #if search_iter > config.regular_ratio * config.search_iter:
                #    config.regular = False

                # sync params from super layer pool
                for i in range(layer_idx, config.layer_num):
                    controller.copy_params_from_super_layer(i)
            
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
                                                {"params": controller.module.super_layers[layer_idx].parameters(), 'lr':config.w_lr},
                                                {"params": controller.module.super_layers[layer_idx+1:].parameters()},
                                                {"params": controller.module.fc_super.parameters()},
                                                {"params": controller.module.distill_aux_head1.parameters()},
                                                {"params": controller.module.distill_aux_head2.parameters()},
                                                {"params": controller.module.ensemble_param},
                                                {"params": controller.module.nas_layers[:layer_idx].parameters()}],
                                                lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
                else:
                    w_optim = torch.optim.SGD([ {"params": controller.module.feature_extractor.parameters()},
                                                {"params": controller.module.super_layers[layer_idx].parameters(), 'lr':config.w_lr},
                                                {"params": controller.module.super_layers[layer_idx+1:].parameters()},
                                                {"params": controller.module.fc_super.parameters()},
                                                {"params": controller.module.distill_aux_head1.parameters()},
                                                {"params": controller.module.distill_aux_head2.parameters()},
                                                {"params": controller.module.nas_layers[:layer_idx].parameters()}],
                                                lr=config.w_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
                # arch_params optimizer
                if config.repeat_cell:
                    alpha_optim = torch.optim.Adam(controller.module.super_layers_arch[0].parameters(), config.alpha_lr, betas=(0.5, 0.999),
                                                weight_decay=config.alpha_weight_decay)
                else:
                    alpha_optim = torch.optim.Adam(controller.module.super_layers_arch[layer_idx:].parameters(), config.alpha_lr, betas=(0.5, 0.999),
                                                weight_decay=config.alpha_weight_decay)
                                                
                if config.ensemble_param:
                    optimizer = torch.optim.SGD([{"params": controller.module.feature_extractor.parameters()},
                                                {"params": controller.module.nas_layers.parameters(), 'lr':config.nasnet_lr*0.1 if config.param_pool_path else config.nasnet_lr},
                                                {"params": controller.module.ensemble_param},
                                                {"params": controller.module.distill_aux_head1.parameters()},
                                                {"params": controller.module.distill_aux_head2.parameters()},
                                                {"params": controller.module.fc_nas.parameters()}],
                                                lr=config.nasnet_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)
                else:
                    optimizer = torch.optim.SGD([{"params": controller.module.feature_extractor.parameters()},
                                                {"params": controller.module.nas_layers.parameters(), 'lr':config.nasnet_lr*0.1 if config.param_pool_path else config.nasnet_lr},
                                                {"params": controller.module.distill_aux_head1.parameters()},
                                                {"params": controller.module.distill_aux_head2.parameters()},
                                                {"params": controller.module.fc_nas.parameters()}],
                                                lr=config.nasnet_lr, momentum=config.w_momentum, weight_decay=config.w_weight_decay)

                lr_scheduler_retrain = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, config.search_iter_epochs, eta_min=config.w_lr_min)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    w_optim, config.search_iter * config.search_iter_epochs, eta_min=config.w_lr_min)

                if (layer_idx == sta_layer_idx) and (resume_state is not None) and (resume_state['sta_search_epoch'] > config.pretrain_epochs):
                    w_optim.load_state_dict(resume_state['w_optim'])
                    alpha_optim.load_state_dict(resume_state['alpha_optim'])
                    lr_scheduler.load_state_dict(resume_state['lr_scheduler'])
                    lr_scheduler_retrain.load_state_dict(resume_state['lr_scheduler_retrain'])
                else:
                    # lr_scheduler 
                    pass
                    #for i in range(search_iter * config.search_iter_epochs):
                    #    lr_scheduler.step()
            
                # warmup model main
                if config.local_rank == 0:
                    logger.info("####### Sub model warmup #######")
                for warmup_epoch in range(config.nasnet_warmup):
                    valid_sampler.set_epoch(warmup_epoch)
                    retrain_warmup(valid_loader, controller, optimizer, layer_idx, warmup_epoch, writer, logger, False, config.nasnet_warmup, config)
                

                best_top1 = 0.
                sub_epoch = 0

                for sub_epoch in range(sta_search_epoch, config.search_iter_epochs):

                    lr_search = lr_scheduler.get_lr()[0]
                    lr_main = lr_scheduler_retrain.get_lr()[0]

                    search_epoch = search_iter * config.search_iter_epochs + sub_epoch

                    # reset iterators
                    train_sampler.set_epoch(search_epoch)
                    valid_sampler.set_epoch(search_epoch)

                    # training
                    search(train_loader, valid_loader, controller, optimizer, w_optim, alpha_optim, layer_idx, search_epoch, writer, logger, config)

                    # validation
                    step_num = len(valid_loader)
                    cur_step = (search_epoch+1) * step_num
                    top1 = 1.

                    genotypes = []
                    connects = []
                    
                    if config.gumbel_sample:
                        genotype, connect = controller.module.generate_genotype_gumbel(0)
                    else:
                        genotype, connect = controller.module.generate_genotype(0)

                    for i in range(config.layer_num):
                        genotypes.append(genotype)
                        connects.append(connect)

                    if config.local_rank == 0:
                        # for i in range(config.layer_num - layer_idx):
                        # logger.info ("Stage: {} Layer: {}".format(layer_idx, i+layer_idx+1))
                        logger.info ("Genotypes: ")
                        # controller.module.print_arch_params(logger, i+layer_idx)
                        controller.module.print_arch_params(logger, 0)

                    for i in range(config.layer_num - layer_idx):
                        if config.local_rank == 0:
                            # genotype
                            genotype = genotypes[i]   
                            logger.info("Stage: {} Layer: {} genotype = {}".format(layer_idx, i+layer_idx+1, genotype))
                            # genotype as a image
                            plot_path = os.path.join(config.plot_path, "Stage_{}_Layer_{}_EP_{:02d}".format(layer_idx, layer_idx+i+1, search_epoch+1))
                            caption = "Stage_{}_Layer_{}_Epoch_{}".format(layer_idx, layer_idx+i+1, search_epoch+1)
                            plot(genotype.normal, plot_path + "-normal", caption)
                            plot(genotype.reduce, plot_path + "-reduce", caption)


                    # sync params to super layer pool
                    for i in range(layer_idx, config.layer_num):
                        controller.module.copy_params_from_nas_layer(i)
                        
                    # save
                    best_top1 = top1
                    best_genotypes = genotypes
                    best_connects = connects

                    for i in range(config.layer_num):
                        controller.module.genotypes[i] = best_genotypes[i]
                        controller.module.connects[i] = best_connects[i]

                    #lr_scheduler.step()
                    #lr_scheduler_retrain.step()
                
            if config.local_rank == 0:
                utils.save_checkpoint(controller.module, config.path, is_best)
                torch.save({
                    'controller': controller.module.state_dict(),
                    'sta_layer_idx': layer_idx,
                    'w_optim': w_optim.state_dict(),
                    'alpha_optim': alpha_optim.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'sta_search_iter': search_iter,
                    'sta_search_epoch': sub_epoch + 1,
                    'best_top1': best_top1,
                    'best_genotypes': best_genotypes,
                    'best_connects': best_connects,
                    'lr_scheduler_retrain': lr_scheduler_retrain.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(config.path, 'search_resume.pth.tar'))


            torch.cuda.empty_cache()
            sta_search_epoch = 0

        # clean
        del w_optim
        del alpha_optim
        del optimizer
        torch.cuda.empty_cache()
        config.pretrain_epochs = max(config.pretrain_epochs - config.pretrain_decay, 0)


    # genotype as a image
    for i in range(config.layer_num):
        genotype, connect = controller.module.generate_genotype(i)
        controller.module.genotypes[i] = genotype
        controller.module.connects[i] = connect

    if config.local_rank == 0:
        for layer_idx, genotype in controller.module.genotypes.items():
            logger.info("layer_idx : {}".format(layer_idx+1))
            logger.info("genotype = {}".format(genotype))

            plot_path = os.path.join(config.plot_path, "Final_Layer_{}_genotype".format(layer_idx+1))
            caption = "Layer_{}".format(layer_idx+1)
            plot(genotype.normal, plot_path + "-normal", caption)
            plot(genotype.reduce, plot_path + "-reduce", caption)


    # save dict as json
    if config.local_rank == 0:
        for layer_idx, genotype in controller.module.genotypes.items():
            controller.module.genotypes[layer_idx] = str(genotype)

        js = json.dumps(controller.module.genotypes)
        file = open('genotypes.json', 'w')  
        file.write(js)  
        file.close()
    
if __name__ == "__main__":
    sta_time = time.time()
    main()
    search_time = time.time() - sta_time
    search_hour = math.floor(search_time / 3600)
    search_min = math.floor(search_time / 60 - search_hour * 60)
    if config.local_rank==0:
        logger.info("Search time: hour: {} minute: {}".format(search_hour, search_min))
