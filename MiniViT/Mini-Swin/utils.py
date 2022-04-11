import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
import argparse
from config import get_config

def parse_option():
    parser = argparse.ArgumentParser('Mini-Swin training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--load_tar', action='store_true', help='whether to load data from tar files')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # Training
    parser.add_argument('--total_train_epoch', default=-1, type=int, help='the total number of epochs for training')
    parser.add_argument('--resume_weight_only', action='store_true', help='whether to only restore weight, used for initialization of multi-stage training')
    parser.add_argument('--base_lr', default=-1.0, type=float, help='the base learning rate')
    parser.add_argument('--weight_decay', default=-1.0, type=float, help='the weight decay value!')
    parser.add_argument('--drop_path_rate', default=-1.0, type=float, help='the value for drop path rate!')
    parser.add_argument('--train_224to384', action='store_true', help='whether finetuning from resolution 224 to 384')

    # MiniViT - Weight Distillation
    parser.add_argument('--do_distill', action='store_true', help='start distillation')
    parser.add_argument('--teacher', default='', type=str, metavar='PATH', help='the path for teacher model')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='the temperature for distillation loss')
    parser.add_argument('--alpha', default=0.0, type=float, help='the weight to balance the soft label loss and ground-truth label loss')
    parser.add_argument('--ar', default=1, type=int, help='The number of relative heads')
    parser.add_argument('--student_layer_list', default='11', type=str, help='The index of layer in the student to be used for distillation loss')
    parser.add_argument('--teacher_layer_list', default='23', type=str, help='The index of layer in the teacher to be used for distillation loss')
    parser.add_argument('--attn_loss', action='store_true', help='whether to use the attention loss')
    parser.add_argument('--hidden_loss', action='store_true', help='whether to use hidden loss along with the attention loss!')
    parser.add_argument('--hidden_weight', default=1.0, type=float, help='the weight for hidden loss!')
    parser.add_argument('--hidden_relation', action='store_true', help='whether to use the hidden relation loss!')
    parser.add_argument('--qkv_weight', default=1.0, type=float, help='the weight for qkv loss!')
    parser.add_argument('--is_student', action='store_true', help='if True, additional linear layers are created for hidden MSE loss')
    parser.add_argument('--fit_size_c', default=-1, type=int, help='when this number is positive, then the output dimension of the linear layers created for hidden MSE loss will be set to this number')

    ## MiniViT - Weight Transformation
    parser.add_argument('--is_sep_layernorm', action='store_true', help='whether to use separate layer normalization in each shared layer')
    parser.add_argument('--is_transform_ffn', action='store_true', help='whether to use transformations for FFN')
    parser.add_argument('--is_transform_heads', action='store_true', help='whether to use transformations for MSA')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    model_state = checkpoint['model']
    if config.TRAIN.TRAIN_224TO384:
        mnames = ['head.weight', 'head.bias']  # (cls, 1024), (cls, )
        now_model_state = model.state_dict()

        if mnames[-1] in model_state:
            ckpt_head_bias = model_state[mnames[-1]]
            if ckpt_head_bias.shape != model.head.bias.shape:
                for mname in mnames:
                    p = model_state[mname].new_zeros(now_model_state[mname].shape)
                    if mname.endswith('.weight'):
                        trunc_normal_(p, std=.02)
                    elif mname.endswith('.bias'):
                        nn.init.constant_(p, 0)
                    else:
                        assert 0

                    model_state[mname] = p

        # drop attn mask
        for k in list(model_state.keys()):
            if 'attn_mask' in k or 'relative_position_index' in k:
                model_state.pop(k)

        mode = 'interpolate'
        for key in list(model_state.keys()):
            value = model_state[key]
            if 'relative_position_bias_table' in key:
                l, nh = value.size()
                l2, nh2 = now_model_state[key].size()
                l2 = int(l2 ** 0.5)
                sl = int(l ** 0.5)
                if sl == 13:
                    pad = 5
                elif sl == 27:
                    pad = 10
                else:
                    assert sl in [23, 47], sl
                    continue

                if mode == "interpolate":
                    # table: (L, num_heads)
                    value = F.interpolate(value.permute(1, 0).view(1, nh, sl, sl),size=(l2, l2), mode='bicubic')  # (1, nh, l2, l2)
                    value = value.reshape(nh, l2 * l2).permute(1, 0)
            model_state[key] = value
    
    if config.TRAIN.TRAIN_224TO384:
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(model_state, strict=True)

    max_accuracy = 0.0
    if config.EVAL_MODE or config.DISTILL.RESUME_WEIGHT_ONLY:
        logger.info(f"==============> RESUME_WEIGHT_ONLY mode is on....................")
        del checkpoint
        torch.cuda.empty_cache()
        return max_accuracy

    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if optimizer is not None and not config.DISTILL.STOP_LOADING_OPTIMIZER:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
            except:
                logger.info('==============> Inconsistency occurred! Skipping loading optimizer...')
        if lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'][0])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    if not config.EVAL_MODE and 'epoch' in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
    if not config.EVAL_MODE and 'epoch' in checkpoint:
        if optimizer is not None and not config.DISTILL.STOP_LOADING_OPTIMIZER:
            try:
                logger.info('Try loading optimizer (2nd trial)')
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info('=> optimizer loaded successfully')
            except:
                logger.info('==============> Inconsistency occurred! Skipping loading optimizer...')

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    if lr_scheduler is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict(),

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt

