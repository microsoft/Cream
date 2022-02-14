""" Retrain cell """
import _init_paths
import os
import torch
import json
import torch.nn as nn
import numpy as np
import lib.utils.genotypes as gt

from tensorboardX import SummaryWriter
from lib.models.cdarts_controller import CDARTSController
from lib.utils import utils
from lib.config import AugmentConfig
from lib.core.augment_function import train, validate

# config
config = AugmentConfig()

# make apex optional
if config.distributed:
    # DDP = torch.nn.parallel.DistributedDataParallel
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex import amp, optimizers
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
if config.local_rank == 0:
    config.print_params(logger.info)
    
if 'cifar' in config.dataset:
    from lib.datasets.cifar import get_augment_datasets
elif 'imagenet' in config.dataset:
    from lib.datasets.imagenet import get_augment_datasets
else:
    raise Exception("Not support dataset!")

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


    loaders, samplers = get_augment_datasets(config)
    train_loader, valid_loader = loaders
    train_sampler, valid_sampler = samplers

    net_crit = nn.CrossEntropyLoss().cuda()
    controller = CDARTSController(config, net_crit, n_nodes=4, stem_multiplier=config.stem_multiplier)

    file = open(config.cell_file, 'r') 
    js = file.read()
    r_dict = json.loads(js)
    if config.local_rank == 0:  
        logger.info(r_dict) 
    file.close()
    genotypes_dict = {}
    for layer_idx, genotype in r_dict.items():
        genotypes_dict[int(layer_idx)] = gt.from_str(genotype)

    controller.build_augment_model(controller.init_channel, genotypes_dict)
    resume_state = None
    if config.resume:
        resume_state = torch.load(config.resume_path,  map_location='cpu')
        controller.model_main.load_state_dict(resume_state['model_main'])

    controller.model_main = controller.model_main.cuda()
    param_size = utils.param_size(controller.model_main)
    logger.info("param size = %fMB", param_size)

    # change training hyper parameters according to cell type
    if 'cifar' in config.dataset:
        if param_size < 3.0:
            config.weight_decay = 3e-4
            config.drop_path_prob = 0.2
        elif param_size > 3.0 and param_size < 3.5:
            config.weight_decay = 3e-4
            config.drop_path_prob = 0.3
        else:
            config.weight_decay = 5e-4
            config.drop_path_prob = 0.3
    
    if config.local_rank == 0:
        logger.info("Current weight decay: {}".format(config.weight_decay))
        logger.info("Current drop path prob: {}".format(config.drop_path_prob))

    controller.model_main = apex.parallel.convert_syncbn_model(controller.model_main)
    # weights optimizer
    optimizer = torch.optim.SGD(controller.model_main.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(controller.model_main.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    if config.use_amp:
        controller.model_main, optimizer = amp.initialize(controller.model_main, optimizer, opt_level=config.opt_level)

    if config.distributed:
        controller.model_main = DDP(controller.model_main, delay_allreduce=True)

    best_top1 = 0.
    best_top5 = 0.
    sta_epoch = 0
    # training loop
    if config.resume:
        optimizer.load_state_dict(resume_state['optimizer'])
        lr_scheduler.load_state_dict(resume_state['lr_scheduler'])
        best_top1 = resume_state['best_top1']
        best_top5 = resume_state['best_top5']
        sta_epoch = resume_state['sta_epoch']

    epoch_pool = [220, 230, 235, 240, 245]
    for epoch in range(sta_epoch, config.epochs):
        # reset iterators
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        current_lr = lr_scheduler.get_lr()[0]
        # current_lr = utils.adjust_lr(optimizer, epoch, config)

        if config.local_rank == 0:
            logger.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < config.warmup_epochs and config.total_batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            if config.local_rank == 0:
                logger.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        
        drop_prob = config.drop_path_prob * epoch / config.epochs
        controller.model_main.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, controller.model_main, optimizer, epoch, writer, logger, config)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1, top5 = validate(valid_loader, controller.model_main, epoch, cur_step, writer, logger, config)

        if 'cifar' in config.dataset:
            lr_scheduler.step()    
        elif 'imagenet' in config.dataset:
            lr_scheduler.step()
            # current_lr = utils.adjust_lr(optimizer, epoch, config)
        else:
            raise Exception('Lr error!')
            
        # save
        if best_top1 < top1:
            best_top1 = top1
            best_top5 = top5
            is_best = True
        else:
            is_best = False

        # save
        if config.local_rank == 0:
            if ('imagenet' in config.dataset) and ((epoch+1) in epoch_pool) and (not config.resume) and (config.local_rank == 0):
                torch.save({
                    "model_main":controller.model_main.module.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "lr_scheduler":lr_scheduler.state_dict(),
                    "best_top1":best_top1,
                    "best_top5":best_top5,
                    "sta_epoch":epoch + 1
                }, os.path.join(config.path, "epoch_{}.pth.tar".format(epoch+1)))
                utils.save_checkpoint(controller.model_main.module.state_dict(), config.path, is_best)

            torch.save({
                "model_main":controller.model_main.module.state_dict(),
                "optimizer":optimizer.state_dict(),
                "lr_scheduler":lr_scheduler.state_dict(),
                "best_top1":best_top1,
                "best_top5":best_top5,
                "sta_epoch":epoch + 1
            }, os.path.join(config.path, "retrain_resume.pth.tar"))
            utils.save_checkpoint(controller.model_main.module.state_dict(), config.path, is_best)
        
    if config.local_rank == 0:
        logger.info("Final best Prec@1 = {:.4%}, Prec@5 = {:.4%}".format(best_top1, best_top5))

    
if __name__ == "__main__":
    main()
