import functools
import logging
import os
import json
import math
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from open_clip.model import convert_to_new_checkpoint, load_pruned_model
from open_clip.factory import load_model, get_tokenizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from open_clip.model import convert_to_new_checkpoint
from open_clip.weight_inherit import weight_inherit

from training.optimizer import build_optimizer


try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, cosine_lr_start, step_lr, cosine_lr_start_nowarmup
from training.train import train_one_epoch, evaluate


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def compute_params(model):
    def _get_params(model):
        if model is None:
            return 0
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        return n_parameters

    def _get_buffers(model):
        if model is None:
            return 0
        n_parameters = sum(p.numel() for p in model.buffers())
        return n_parameters

    n_parameters = _get_params(model)
    num_params_image = _get_params(model.image_encoder_without_ddp.visual)
    num_buffers_image = _get_buffers(model.image_encoder_without_ddp.visual)
    num_params_text = _get_params(model.text_encoder_without_ddp.transformer)
    num_token_emb = _get_params(model.text_encoder_without_ddp.token_embedding) if \
        model.text_encoder_without_ddp.transformer is not None else 0
    if model.text_encoder_without_ddp.transformer is not None and \
            sum(p.numel() for p in model.text_encoder_without_ddp.transformer.parameters()) > 0:
        num_params_text += _get_params(
            model.text_encoder_without_ddp.token_embedding)
        num_params_text += _get_params(model.text_encoder_without_ddp.ln_final)
        num_params_text += (model.text_encoder_without_ddp.positional_embedding.numel() +
                            model.text_encoder_without_ddp.text_projection.numel())
    return n_parameters, (num_params_image, num_buffers_image), num_params_text, num_token_emb


DEVICE = torch.device('cpu')


def _load_checkpoint(name):
    global DEVICE
    if '@' in name:
        teacher_model_name, teacher_pretrained = name.split('@')
        _model, _, _ = create_model_and_transforms(
            teacher_model_name, pretrained=teacher_pretrained, device=DEVICE)
        return _model.state_dict()
    json_fname = os.path.join('exps', name + '.json')
    if os.path.exists(json_fname):
        model_info = json.load(open(json_fname))
        name = model_info['resume']
    state_dict = torch.load(name, map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    return state_dict


def main():
    global DEVICE
    args = parse_args()

    is_bf16_supported = torch.cuda.is_bf16_supported()
    if not is_bf16_supported:
        for name in ['precision', 'image_precision', 'text_precision', 'logit_precision']:
            if getattr(args, name) == 'amp_bfloat16':
                setattr(args, name, 'amp')

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if False and os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    DEVICE = device

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(
            args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(
            args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    assert args.precision in ['amp', 'amp_bfloat16', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        # the model will be converted to FP16 if args.precision is fp16
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        args=args,
    )
    random_seed(args.seed, args.rank)

    if is_master(args, local=args.log_local):
        logging.info('train: {}\n val: {}'.format(
            preprocess_train, preprocess_val))

    n_parameters, (num_params_image,
                   num_buffers_image), num_params_text, num_token_emb = compute_params(model)
    if is_master(args):
        logging.info(f"number of params: {n_parameters / 1e6}")
        logging.info(f'number of params image: {num_params_image / 1e6}')
        logging.info(f'number of buffers image: {num_buffers_image / 1e6}')
        logging.info(f'number of params text: {num_params_text / 1e6}')
        logging.info(
            f'number of token embedding in text encoder : {num_token_emb / 1e6}')

    if args.distillation:
        teacher_model = load_model(args.distillation_teacher, device=device)

        if args.grad_checkpointing:
            teacher_model.set_grad_checkpointing()
        teacher_model.eval()
        teacher_model.cuda()
        # frozen parameters
        for p in teacher_model.parameters():
            p.requires_grad = False

        model.teacher = [teacher_model]
    else:
        teacher_model = None

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
        logging.info('Locked image tower.')

    if args.lock_text:
        model.lock_text_tower()
        logging.info('Locked text tower.')

    model.cuda()

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    model_without_ddp = model

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        optimizer = build_optimizer(args, model)
        assert not args.horovod

        use_loss_scale = any(map(
            lambda x: x in ['amp', 'fp16'],
            [args.precision, args.image_precision, args.text_precision, args.logit_precision]))
        print(f'Use loss scale: {use_loss_scale}')
        scaler = GradScaler(enabled=use_loss_scale)

    checkpoint_fname_list = [None]
    if is_master(args):
        if os.path.isdir(args.checkpoint_path):
            ckpts_list = []
            for name in os.listdir(args.checkpoint_path):
                if name.startswith('epoch_') and name.endswith('.pt'):
                    name = os.path.splitext(name)[0]
                    name = name[len('epoch_'):]
                    epoch, it = map(int, name.split('_iter_'))
                    ckpts_list.append((epoch, it))
            if len(ckpts_list) > 0:
                ckpts_list.sort(reverse=True)
                for epoch, it in ckpts_list:
                    checkpoint_fname = os.path.join(
                        args.checkpoint_path, f"epoch_{epoch}_iter_{it}.pt")
                    try:
                        # check valid
                        torch.load(checkpoint_fname, map_location='cpu')
                        checkpoint_fname_list[0] = checkpoint_fname
                        break
                    except Exception as e:
                        print(f'Load Ckpt Fail: {e}')
    torch.distributed.broadcast_object_list(checkpoint_fname_list, src=0)

    if checkpoint_fname_list[0] is not None:
        print(
            f'overwrite checkpoint path: {checkpoint_fname_list[0]}, the original path is {args.resume}')
        args.resume = checkpoint_fname_list[0]

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    start_epoch = 0

    # optionally resume from a checkpoint
    start_epoch = 0
    start_iter = 0
    if args.resume is not None:
        # this part only suppots resume clip model without mask. [TODO]: support resume clip model with mask.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if args.prune_image and args.prune_text:
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                sd = {k.replace('.module', ''): v for k, v in sd.items()}
                logging.info('convert pruned model to base')
                load_pruned_model(model, sd)

                if args.load_last_stage is False:
                    logging.info('=== FUSE MASK IMAGE ===')
                    num_params_before_fuse = sum(
                        p.numel() for p in model.image_encoder_without_ddp.parameters() if p.requires_grad)
                    with torch.no_grad():
                        model.image_encoder_without_ddp.eval()
                        image = torch.randn((1, 3, 224, 224), device='cuda')
                        model.image_encoder_without_ddp(image)
                        model.image_encoder_without_ddp = model.image_encoder_without_ddp.prune()
                    assert hasattr(
                        model.image_encoder_without_ddp, 'l0_module')
                    model.image_encoder_without_ddp.l0_module = None
                    num_params_after_fuse = sum(
                        p.numel() for p in model.image_encoder_without_ddp.parameters() if p.requires_grad)
                    logging.info(
                        f'=> fuse MASK image: {num_params_before_fuse} -> {num_params_after_fuse}')

                    logging.info('=== FUSE MASK TEXT ===')
                    num_params_before_fuse = sum(
                        p.numel() for p in model.text_encoder_without_ddp.parameters() if p.requires_grad)
                    with torch.no_grad():
                        model.text_encoder_without_ddp.eval()
                        text = torch.randint(0, 100, (1, 77), device='cuda')
                        model.text_encoder_without_ddp(text)
                        model.text_encoder_without_ddp = model.text_encoder_without_ddp.prune()
                    assert hasattr(model.text_encoder_without_ddp, 'l0_module')
                    model.text_encoder_without_ddp.l0_module = None
                    num_params_after_fuse = sum(
                        p.numel() for p in model.text_encoder_without_ddp.parameters() if p.requires_grad)
                    logging.info(
                        f'=> fuse MASK text: {num_params_before_fuse} -> {num_params_after_fuse}')
                    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
            else:
                sd = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in sd.items():
                    if 'logit_scale' in key:
                        new_key = '_logit_scale.logit_scale'
                    elif key.startswith('module.visual'):
                        new_key = key.replace(
                            'module.visual', '_image_encoder.visual')
                    elif key.startswith('module'):
                        new_key = key.replace('module', '_text_encoder')
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                sd = new_state_dict
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)

            if 'epoch' in checkpoint and args.load_last_stage is False:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]

            if optimizer is not None and 'optimizer' in checkpoint and args.load_last_stage is False:
                if len(optimizer) == len(checkpoint['optimizer']):
                    for opt, v in zip(optimizer, checkpoint["optimizer"]):
                        assert len(opt.param_groups) == len(v['param_groups']), \
                            f'number of param groups mismatch: {len(opt.param_groups)} vs {len(v["param_groups"])}'
                        opt.load_state_dict(v)
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                else:
                    logging.info(f"optimizer load fails, use new one")

            if 'iter_in_epoch' in checkpoint and args.load_last_stage is False:
                start_iter = checkpoint['iter_in_epoch'] + 1
                logging.info(f"fast_forward dataloader to iter {start_iter}")

        else:
            raise FileNotFoundError(f'=> no checkpoint found at {args.resume}')
    else:

        def remove_prefix_module(state_dict):
            # remove the first or the second module
            return convert_to_new_checkpoint(state_dict)

        def add_prefix_module(state_dict):
            if all(map(lambda x: not x.startswith('module.'), state_dict.keys())):
                return {'module.' + k: v for k, v in state_dict.items()}
            return state_dict

        def model_load_checkpoint(model, state_dict):
            if hasattr(model, 'module'):
                state_dict = add_prefix_module(state_dict)
            model.load_state_dict(state_dict, strict=True)

        def encoder_weight_inherit(student_state, teacher_state, encoder_prefix, head_dim):
            def _filter_prefix(state, prefix):
                return dict((k, v) for k, v in state.items() if k.startswith(prefix) and 'l0_module' not in k)
            student_fs = _filter_prefix(student_state, encoder_prefix)
            teacher_fs = _filter_prefix(teacher_state, encoder_prefix)
            logging.info(
                f'  student: {len(student_fs)}, teacher: {len(teacher_fs)}')
            weight_inherit(student_fs, teacher_fs, head_dim)
            num = 0
            for k, v in student_fs.items():
                num += v.numel()
                student_state[k] = v
            return num

        if args.pretrained_image_file:
            logging.info('=== INHERIT IMAGE ===')
            # no resume, try to load image file
            state_dict = remove_prefix_module(model.state_dict())
            # ckpt
            image_checkpoint = remove_prefix_module(
                _load_checkpoint(args.pretrained_image_file))
            num_inherit = encoder_weight_inherit(
                state_dict, image_checkpoint, '_image_encoder.visual', head_dim=model.visual.head_dim)
            # format: _image_encoder.xxxx
            model_load_checkpoint(model, state_dict)
            assert num_inherit == num_params_image + \
                num_buffers_image, (num_inherit,
                                    num_params_image, num_buffers_image)
            logging.info(
                f'=> loaded image checkpoint {args.pretrained_image_file} ({num_inherit} image params)')

        if args.pretrained_text_file:
            logging.info('=== INHERIT TEXT ===')
            # student with ddp
            state_dict = remove_prefix_module(model.state_dict())
            # teacher without ddp
            text_checkpoint = remove_prefix_module(
                _load_checkpoint(args.pretrained_text_file))
            # format: _text_encoder.xxxx
            num_inherit = encoder_weight_inherit(
                state_dict, text_checkpoint, '_text_encoder', head_dim=model.transformer.head_dim)
            assert num_inherit == num_params_text, (
                num_inherit, num_params_text)
            logging.info(
                f'=> loaded text checkpoint {args.pretrained_text_file} ({num_inherit} text params)')
            model_load_checkpoint(model, state_dict)

    if args.distributed and not args.horovod:
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        ddp_fn = functools.partial(
            torch.nn.parallel.DistributedDataParallel, device_ids=[device], **ddp_args)
        # re-ddpify
        model.ddpify(ddp_fn)

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val),
                    epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    print(f"Dataset: {set(data.keys())}")
    assert len(data), 'At least one train or eval dataset must be specified.'

    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb_output_path = args.checkpoint_path
        wandb.init(
            project="tinyclip",
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            dir=wandb_output_path,
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        if args.prune_image or args.prune_text:
            scheduler = cosine_lr(
                optimizer[0:3], args.lr, args.prune_step, total_steps)
            scheduler_l0 = step_lr(optimizer[-1], args.prune_step)
        else:
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
            scheduler_l0 = None

    if 'train' not in data or args.eval:
        results = evaluate(model, data, start_epoch, args, writer)
        if is_master(args):
            print(results)
        return

    for epoch in range(start_epoch, math.ceil(args.epochs)):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        rtn = train_one_epoch(model, data, epoch, optimizer, scaler,
                              scheduler, scheduler_l0, args, writer, start_iter)
        if isinstance(rtn, str) and rtn == 'non-finite loss':
            break
        else:
            model, optimizer, scaler, scheduler, scheduler_l0, args = rtn
        start_iter = 0

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if False and os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path,
             ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
