import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from AutoFormer.experiments.super_resolution.supernet_engine import train_one_epoch, evaluate, sample_configs_swinir
from AutoFormer.lib import utils
from AutoFormer.lib.config import cfg, update_config_from_file
from AutoFormer.utils import utils_option as option
# from models.model_plain import ModelPlain
from torch.utils.data import DataLoader
from AutoFormer.data.dataset_sr import DatasetSR
from torch.utils.data.distributed import DistributedSampler
from AutoFormer.model.swinIR.network_swinir import SwinIR


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file

    parser.add_argument('--cfg', help='experiment configure file name', type=str,
                        default='./experiments_configs/supernet-swinir/supernet-T.yaml')

    parser.add_argument('--opt-doc', type=str, default='./experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json',
                        help='Path to option JSON file for SwinIR.')

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14,
                        help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'retrain'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=64, type=int)
    parser.add_argument('--patch_size', default=8, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.999, metavar='M',
                        help='SGD momentum (default: 0.999)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='step (default: "step"')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limiThis should bet percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--lr-power', type=float, default=1.0,
    #                     help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=50, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.5)')
    # parser.add_argument('--decay-milestones', '--dmile', type=list, default=[50, 100, 150, 200, 250], metavar='MILESTONES',
    #                     help='LR decay milestones (default: [50, 100, 150, 200, 250])')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./output_swinir_final_split',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    # For reading from .json file, major part of .json can be turned into commandline args
    opt = option.parse(parser.parse_args().opt_doc, is_train=True)

    border = opt['scale']
    args.scale = border

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = DatasetSR(dataset_opt)
            print('Dataset [{:s} - {:s}] is created.'.format(train_set.__class__.__name__, dataset_opt['name']))
            print(dataset_opt)
            if args.distributed:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                # train_sampler = DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True, seed=seed)
                train_sampler = DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank,
                                                   shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
            else:
                train_sampler = torch.utils.data.RandomSampler(train_set)
            data_loader_train = DataLoader(
                train_set, sampler=train_sampler,
                batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                pin_memory=args.pin_mem,
                drop_last=True,
            )
        elif phase == 'valid' and args.mode == 'super':
            test_set = DatasetSR(dataset_opt)
            print('Dataset [{:s} - {:s}] is created.'.format(test_set.__class__.__name__, dataset_opt['name']))
            print(dataset_opt)
            data_loader_test = DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=8,
                                          drop_last=False, pin_memory=True)

        elif phase == 'test' and args.mode == 'retrain':
            test_set = DatasetSR(dataset_opt)
            print('Dataset [{:s} - {:s}] is created.'.format(test_set.__class__.__name__, dataset_opt['name']))
            print(dataset_opt)
            data_loader_test = DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=4,
                                          drop_last=False, pin_memory=True)

    print(f"Creating SwinIRTransformer")
    print(cfg)
    opt_net = opt['netG']
    model = SwinIR(img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   depths=cfg.SUPERNET.DEPTHS,
                   embed_dim=cfg.SUPERNET.EMBED_DIM,
                   num_heads=cfg.SUPERNET.NUM_HEADS,
                   mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                   upsampler=opt_net['upsampler'],
                   upscale=border)

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM, 'rstb_num': cfg.SEARCH_SPACE.RSTB_NUM,
               'stl_num': cfg.SEARCH_SPACE.STL_NUM}

    model.to(device)

    teacher_model = None
    teacher_loss = None
    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # TODO: Adapt criterion based on the arg in json file 
    criterion = torch.nn.L1Loss()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments_configs
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    retrain_config = None
    if args.mode == 'retrain' and "RETRAIN" in cfg:
        retrain_config = {'layer_num': cfg.RETRAIN.DEPTH, 'embed_dim': [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
                          'num_heads': cfg.RETRAIN.NUM_HEADS, 'mlp_ratio': cfg.RETRAIN.MLP_RATIO}
    if args.eval:
        test_stats = evaluate(data_loader_test, model, device, mode=args.mode, retrain_config=retrain_config,
                              scaling=args.scale)
        print(f"PSNR of the network on the {len(test_set)} test images: {test_stats['psnr']:.1f}%")
        return

    print("Start training")
    start_time = time.time()
    max_psnr = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, None,
            amp=args.amp, teacher_model=teacher_model,
            teach_loss=teacher_loss,
            choices=choices, mode=args.mode, retrain_config=retrain_config,
            sampler=sample_configs_swinir,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_test, model, device, amp=args.amp, choices=choices, mode=args.mode,
                              retrain_config=retrain_config)
        max_psnr = max(max_psnr, test_stats["psnr"])
        print(f'Max psnr: {max_psnr:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
