""" Config class for search/augment """
import argparse
import os
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        ########### basic settings ############
        parser.add_argument('--dataset', default='imagenet', help='CIFAR10 / MNIST / FashionMNIST / imagenet')
        parser.add_argument('--model_type', type=str, default='cifar', help='cifar or imagenet')
        parser.add_argument('--data_dir', type=str, default='experiments/data/cifar', help='cifar dataset')
        parser.add_argument('--train_dir', type=str, default='experiments/data/imagenet/train', help='')
        parser.add_argument('--val_dir', type=str, default='experiments/data/imagenet/train', help='')
        parser.add_argument('--test_dir', type=str, default='experiments/data/imagenet/val', help='')
        parser.add_argument('--param_pool_path', type=str, default=None, help='')
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--stem_multiplier', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--sample_ratio', type=float, default=0.2, help='imagenet sample ratio')
        parser.add_argument('--resume', action='store_true', default=False, help='resnet stem(pretrain)')

        ########### learning rate ############
        parser.add_argument('--w_lr', type=float, default=0.05, help='lr for weights')
        parser.add_argument('--lr_ratio', type=float, default=0.5, help='lr for trained layers')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--alpha_lr', type=float, default=6e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        ########### alternate training ############
        parser.add_argument('--res_stem', action='store_true', default=False, help='resnet stem(pretrain)')
        parser.add_argument('--layer_num', type=int, default=3, help='layer need to be replaced')
        parser.add_argument('--cells_num', type=int, default=3, help='cells num of one layer')
        parser.add_argument('--pretrain_epochs', type=int, default=5, help='# of training epochs')
        parser.add_argument('--pretrain_decay', type=int, default=5, help='pretrain epochs')
        parser.add_argument('--random_times', type=int, default=10, help='# of training epochs')
        parser.add_argument('--random_epochs', type=int, default=3, help='# of training epochs')
        parser.add_argument('--search_iter', type=int, default=5, help='times of search')
        parser.add_argument('--search_iter_epochs', type=int, default=5, help='# of training epochs')
        parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
        parser.add_argument('--one_stage', action='store_true', default=False, help='one_stage search')
        parser.add_argument('--same_structure', action='store_true', default=False, help='same_structure search and retrain')
        parser.add_argument('--clean_arch', action='store_true', default=False, help='clean archs each epoch')
        parser.add_argument('--sync_param', action='store_true', default=False, help='whether to sync param')
        parser.add_argument('--ensemble_sum', action='store_true', default=False, help='ensemble sum or concat')
        parser.add_argument('--ensemble_param', action='store_true', default=False, help='whether to learn ensemble params')
        parser.add_argument('--use_beta', action='store_true', default=False, help='whether to use beta arch param')
        parser.add_argument('--bn_affine', action='store_true', default=False, help='main bn affine')
        parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to sync bn')
        parser.add_argument('--use_apex', action='store_true', default=False, help='whether to apex')
        parser.add_argument('--regular', action='store_true', default=False, help='resnet stem(pretrain)')
        parser.add_argument('--regular_ratio', type=float, default=0.5, help='regular ratio')
        parser.add_argument('--regular_coeff', type=float, default=5, help='regular coefficient')
        parser.add_argument('--repeat_cell', action='store_true', default=False, help='use repeat cell')
        parser.add_argument('--fix_head', action='store_true', default=False, help='whether to fix head')
        parser.add_argument('--share_fc', action='store_true', default=False, help='whether to share fc')
        parser.add_argument('--nasnet_lr', type=float, default=0.1, help='lr of nasnet')
        parser.add_argument('--nasnet_warmup', type=int, default=5, help='warm up of nasnet')
        parser.add_argument('--loss_alpha', type=float, default=1, help='loss alpha')
        parser.add_argument('--loss_T', type=float, default=2, help='loss T')
        parser.add_argument('--interactive_type', type=int, default=0, help='0 kl 1 cosine 2 mse 3 sl1')
        parser.add_argument('--gumbel_sample', action='store_true', default=False, help='whether to use gumbel sample')
        parser.add_argument('--sample_pretrain', action='store_true', default=False, help='sample_pretrain')


        ########### data augument ############
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
        parser.add_argument('--mixup_alpha', default=0., type=float,
                            help='mixup interpolation coefficient (default: 1)')

        ########### distributed ############
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument("--world_size", default=1, type=int)
        parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--distributed', action='store_true', help='Run model distributed mode.')


        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './experiments/data/'
        self.path = os.path.join('experiments', 'search', self.name)
        self.resume_path = os.path.join(self.path, 'search_resume.pth.tar')
        self.plot_path = os.path.join(self.path, 'plots')
        self.retrain_path = os.path.join(self.path, 'retrain')
        self.gpus = parse_gpus(self.gpus)


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='cifar10 / cifar100 / imagenet')
        parser.add_argument('--model_type', type=str, default='cifar', help='cifar or imagenet')

        parser.add_argument('--data_dir', type=str, default='experiments/data/cifar', help='cifar dataset')
        parser.add_argument('--train_dir', type=str, default='experiments/data/imagenet/train', help='')
        parser.add_argument('--test_dir', type=str, default='experiments/data/imagenet/val', help='')
        parser.add_argument('--cell_file', type=str, default='CDARTS/cells/cifar_genotype.json', help='')
        parser.add_argument('--resume', action='store_true', default=False, help='resnet stem(pretrain)')

        parser.add_argument('--n_classes', type=int, default=10)
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--stem_multiplier', type=int, default=3)

        ########### alternate training ############
        parser.add_argument('--res_stem', action='store_true', default=False, help='resnet stem(pretrain)')
        parser.add_argument('--layer_num', type=int, default=3, help='layer need to be replaced')
        parser.add_argument('--cells_num', type=int, default=3, help='cells num of one layer')
        parser.add_argument('--same_structure', action='store_true', default=False, help='same_structure search and retrain')
        parser.add_argument('--ensemble_sum', action='store_true', default=False, help='whether to ensemble')
        parser.add_argument('--ensemble_param', action='store_true', default=False, help='whether to learn ensemble params')
        parser.add_argument('--use_beta', action='store_true', default=False, help='whether to use beta arch param')
        parser.add_argument('--bn_affine', action='store_true', default=False, help='main bn affine')
        parser.add_argument('--repeat_cell', action='store_true', default=False, help='use repeat cell')
        parser.add_argument('--fix_head', action='store_true', default=False, help='whether to fix head')
        parser.add_argument('--share_fc', action='store_true', default=False, help='whether to share fc')
        parser.add_argument('--sample_pretrain', action='store_true', default=False, help='sample_pretrain')

        parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
        parser.add_argument('--mixup_alpha', default=0., type=float,
                            help='mixup interpolation coefficient (default: 1)')
        parser.add_argument('--resume_name', type=str, default='retrain_resume.pth.tar')

        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--warmup_epochs', type=int, default=5, help='# warmup')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--sample_archs', type=int, default=1, help='sample arch num')
        parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
        parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path prob')

        ########### distributed ############
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument("--world_size", default=1, type=int)
        parser.add_argument('--use_amp', action='store_true', default=False, help='whether to use amp')
        parser.add_argument('--opt-level', type=str, default='O1')

        parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
        parser.add_argument('--distributed', action='store_true',
                    help='Run model distributed mode.')

        parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
        parser.add_argument('--dynamic-loss-scale', action='store_true',
                            help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                            '--static-loss-scale.')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './experiments/data/'
        self.path = os.path.join('experiments', 'retrain', self.name)
        self.gpus = parse_gpus(self.gpus)
        self.resume_path = os.path.join(self.path, self.resume_name)

