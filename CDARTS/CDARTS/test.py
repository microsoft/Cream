""" Search cell """
import _init_paths
import os
import torch
import json
import numpy as np
import lib.utils.genotypes as gt

from tensorboardX import SummaryWriter
from lib.models.model_test import ModelTest
from lib.utils import utils
from lib.config import AugmentConfig
from lib.core.augment_function import validate

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
    raise Exception("Not support dataser!")

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

    file = open(config.cell_file, 'r') 
    js = file.read()
    r_dict = json.loads(js)
    if config.local_rank == 0:  
        logger.info(r_dict) 
    file.close()
    genotypes_dict = {}
    for layer_idx, genotype in r_dict.items():
        genotypes_dict[int(layer_idx)] = gt.from_str(genotype)

    model_main = ModelTest(genotypes_dict, config.model_type, config.res_stem, init_channel=config.init_channels, \
                            stem_multiplier=config.stem_multiplier, n_nodes=4, num_classes=config.n_classes)
    resume_state = torch.load(config.resume_path,  map_location='cpu')
    model_main.load_state_dict(resume_state, strict=False)
    model_main = model_main.cuda()

    if config.distributed:
        model_main = DDP(model_main, delay_allreduce=True)

    top1, top5 = validate(valid_loader, model_main, 0, 0, writer, logger, config)
    if config.local_rank == 0:
        print("Final best Prec@1 = {:.4%}, Prec@5 = {:.4%}".format(top1, top5))

if __name__ == "__main__":
    main()
