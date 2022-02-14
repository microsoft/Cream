import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.genotypes as gt
import logging
import copy

from lib.models import ops
from lib.models.search_cells import SearchCell
from lib.models.augment_cells import AugmentCell
from lib.models.aux_head import AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet, DistillHeadCIFAR, DistillHeadImagenet
from lib.models.model_augment import ModelAug

class CDARTSController(nn.Module):
    """ CDARTS Controller"""
    def __init__(self, config, criterion, n_nodes=4, stem_multiplier=3, genotypes={}):
        """
        args:

        """
        super(CDARTSController, self).__init__()

        # some settings
        self.n_nodes = n_nodes
        self.n_ops = len(gt.PRIMITIVES)
        self.criterion = criterion
        self.layer_num = config.layer_num
        self.c_in = config.input_channels
        self.num_classes = config.n_classes
        # cifar10 or imagenet
        self.model_type = config.model_type
        self.stem_multiplier = stem_multiplier
        self.init_channel = config.init_channels
        self.res_stem = config.res_stem
        self.ensemble_sum = config.ensemble_sum
        self.use_ensemble_param = config.ensemble_param
        self.use_beta = config.use_beta
        self.bn_affine = config.bn_affine
        self.repeat_cell = config.repeat_cell
        self.fix_head = config.fix_head
        self.share_fc = config.share_fc
        self.sample_pretrain = config.sample_pretrain

        if self.model_type == 'cifar':
            self.layers = [3, 3, 2]
            self.layers_reduction = [True, True, False]
            self.augment_layers = [7, 7, 6]
            self.nas_layers = nn.ModuleList([None, None, None])

        elif self.model_type == 'imagenet':
            if self.res_stem:
                self.layers = [2, 2, 2, 2]
                self.nas_layers = nn.ModuleList([None, None, None, None])
                self.layers_reduction = [False, True, True, True]
                self.augment_layers = [3, 4, 3, 4]
            else:
                self.layers = [3, 3, 2]
                self.nas_layers = nn.ModuleList([None, None, None])
                self.layers_reduction = [True, True, False]
                self.augment_layers = [5, 5, 4]
        else:
            raise Exception("Wrong model type!")

        # use genotypes to generate search layers
        self.genotypes = genotypes
        self.connects = {}
        self.fc_super = None
        self.fc_nas = None
        self.distill_aux_c1 = None
        self.distill_aux_c2 = None
        self.feature_extractor = None
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.super_layers = nn.ModuleList()
        self.super_layers_arch = nn.ModuleList()

        self.super_layers_pool = nn.ModuleList()
        self.super_layers_pool_arch = nn.ModuleList()
        self.model_main = None

        self.build_init_model()

    ######################## ---------------------------- ########################
    ######################## Functions for update modules ########################
    ######################## ---------------------------- ########################
    def build_init_model(self):
        self.extractor_grad = True
        if self.model_type == 'cifar':
            self.feature_extractor = self.cifar_stem(self.init_channel * self.stem_multiplier)
            reduction_p = False
        elif self.model_type == 'imagenet':
            if self.res_stem:
                self.feature_extractor = self.resnet_stem(self.init_channel * self.stem_multiplier)
                reduction_p = False
            else:
                self.feature_extractor = self.imagenet_stem(self.init_channel * self.stem_multiplier)
                reduction_p = True
        else:
            raise Exception("error! not support now!")

        c_p = self.init_channel * self.stem_multiplier
        c_pp = self.init_channel * self.stem_multiplier
        c_cur = self.init_channel
        self.super_layers_pool_arch.append(self.pretrain_architecture_params(self.n_ops))

        if self.repeat_cell:
            self.super_layers_arch.append(self.add_architecture_params(self.n_ops))

        for layer_idx in range(self.layer_num):
            reduction = self.layers_reduction[layer_idx]
            
            super_layer = self.add_super_layer(c_cur, c_p, c_pp, reduction_p, reduction, self.layers[layer_idx])
            super_layer_pool = self.add_super_layer(c_cur, c_p, c_pp, reduction_p, reduction, self.augment_layers[layer_idx], is_slim=self.sample_pretrain)
            super_layer_arch = self.add_architecture_params(self.n_ops)

            self.freeze_unused_params(super_layer_arch, reduction, self.layers[layer_idx])
            self.super_layers.append(super_layer)
            self.super_layers_pool.append(super_layer_pool)
            if not self.repeat_cell:
                self.super_layers_arch.append(super_layer_arch)

            if reduction:
                c_p = c_cur * 2 * self.n_nodes
            else:
                c_p = c_cur * self.n_nodes

            if self.res_stem:
                c_pp = c_p
                reduction_p = False
            else:
                c_pp = c_cur * self.n_nodes
                reduction_p = reduction

            if layer_idx == self.layer_num-3:
                self.distill_aux_c1 = c_p
            if layer_idx == self.layer_num-2:
                self.distill_aux_c2 = c_p

            if reduction:
                c_cur = c_cur * 2
            else:
                c_cur = c_cur

        self.fc_super = nn.Linear(c_p, self.num_classes)
        if self.share_fc:
            self.fc_nas = self.fc_super
        else:
            self.fc_nas = nn.Linear(c_p, self.num_classes)

        if self.use_ensemble_param:
            self.ensemble_param = nn.Parameter(0.333*torch.rand(3), requires_grad=True)
        else:
            self.ensemble_param = nn.Parameter(0.333*torch.ones(3), requires_grad=False)
        if self.model_type == 'cifar':
            self.distill_aux_head1 = DistillHeadCIFAR(self.distill_aux_c1, 6, self.num_classes, bn_affine=False)
            self.distill_aux_head2 = DistillHeadCIFAR(self.distill_aux_c2, 6, self.num_classes, bn_affine=False)
        elif self.model_type == 'imagenet':
            if self.res_stem:
                self.distill_aux_head1 = DistillHeadImagenet(self.distill_aux_c1, 14, self.num_classes, bn_affine=False)
                self.distill_aux_head2 = DistillHeadImagenet(self.distill_aux_c2, 6, self.num_classes, bn_affine=False)
            else:
                self.distill_aux_head1 = DistillHeadImagenet(self.distill_aux_c1, 6, self.num_classes, bn_affine=False)
                self.distill_aux_head2 = DistillHeadImagenet(self.distill_aux_c2, 5, self.num_classes, bn_affine=False)
        else:
            raise Exception("error! not support now!")


        self.fix_structure()

    def fix_structure(self):
        if self.fix_head:
            for n, p in self.distill_aux_head1.named_parameters():
                p.requires_grad = False
            for n, p in self.distill_aux_head2.named_parameters():
                p.requires_grad = False

    def fix_pre_layers(self, layer_idx=0):
        for i in range(layer_idx):
            for name, param in self.super_layers_arch[i].named_parameters():
                param.requires_grad=False

    def build_nas_layers(self, layer_idx, best_genotype, same_structure=False):
        c_p = self.init_channel * self.stem_multiplier
        c_pp = self.init_channel * self.stem_multiplier
        c_cur = self.init_channel
        if self.model_type == 'cifar':
            reduction_p = False
        elif self.model_type == 'imagenet':
            if self.res_stem:
                reduction_p = False
            else:
                reduction_p = True
        else:
            raise Exception("error! not support now!") 

        for i in range(self.layer_num):
            reduction = self.layers_reduction[i]

            if i == layer_idx:
                break

            if reduction:
                c_p = c_cur * 2 * self.n_nodes
            else:
                c_p = c_cur * self.n_nodes

            if self.res_stem:
                c_pp = c_p
                reduction_p = False
            else:
                c_pp = c_cur * self.n_nodes
                reduction_p = reduction

            if reduction:
                c_cur = c_cur * 2
            else:
                c_cur = c_cur

        # once model search is well trained, transfor model params from model_search to model_main
        # genotype = self.generate_genotype(self.model_search.arch_params)
        if same_structure:
            nas_layer = self.generate_nas_layer(c_cur, c_p, c_pp, reduction_p, reduction, best_genotype, self.layers[layer_idx], bn_affine=self.bn_affine)
        else:
            nas_layer = self.generate_nas_layer(c_cur, c_p, c_pp, reduction_p, reduction, best_genotype, self.augment_layers[layer_idx], bn_affine=self.bn_affine)
        self.genotypes[layer_idx] = best_genotype
        self.nas_layers[layer_idx] = nas_layer

    def build_augment_model(self, init_channel, genotypes_dict):
        if len(genotypes_dict.keys()) == 0:
            raise Exception("error! genotypes is empty!")
        else:
            self.extractor_grad = True
            if self.model_type == 'cifar':
                feature_extractor = self.cifar_stem(self.init_channel * self.stem_multiplier)
                reduction_p = False
            elif self.model_type == 'imagenet':
                if self.res_stem:
                    feature_extractor = self.resnet_stem(self.init_channel * self.stem_multiplier)
                    reduction_p = False
                else:
                    feature_extractor = self.imagenet_stem(self.init_channel * self.stem_multiplier)
                    reduction_p = True
            else:
                raise Exception("error! not support now!")

            c_p = self.init_channel * self.stem_multiplier
            c_pp = self.init_channel * self.stem_multiplier
            c_cur = self.init_channel

            for layer_idx, genotype in genotypes_dict.items():
                reduction = self.layers_reduction[layer_idx]
                nas_layer = self.generate_nas_layer(c_cur, c_p, c_pp, reduction_p, reduction, genotype, self.augment_layers[layer_idx])
                self.nas_layers[layer_idx] = nas_layer
                
                if reduction:
                    c_p = c_cur * 2 * self.n_nodes
                else:
                    c_p = c_cur * self.n_nodes

                if self.res_stem:
                    c_pp = c_p
                    reduction_p = False
                else:
                    c_pp = c_cur * self.n_nodes
                    reduction_p = reduction

                if reduction:
                    c_cur = c_cur * 2
                else:
                    c_cur = c_cur

                if layer_idx == self.layer_num-2:
                    c_aux = c_p

            if self.model_type == 'cifar':
                aux_head = AuxiliaryHeadCIFAR(c_aux, 5, self.num_classes)
            elif self.model_type == 'imagenet':
                if self.res_stem:
                    aux_head = AuxiliaryHeadImageNet(c_aux, 12, self.num_classes)
                else:
                    aux_head = AuxiliaryHeadImageNet(c_aux, 5, self.num_classes)
            else:
                aux_head = None

            # super_layers = copy.deepcopy(self.super_layers)
            # super_layers_arch = copy.deepcopy(self.super_layers_arch)
            nas_layers = copy.deepcopy(self.nas_layers)
            fc = copy.deepcopy(self.fc_nas)
            self.model_main = ModelAug(feature_extractor, nas_layers, fc, n_nodes=self.n_nodes, aux_head=aux_head)
 
    def freeze_unused_params(self, super_layer_arch, reduction, cell_num):
        if not reduction:
            for name, param in super_layer_arch.named_parameters():
                if name.startswith('1') or name.startswith('3'):
                    param.requires_grad=False
        elif cell_num == 1 and reduction:
            for name, param in super_layer_arch.named_parameters():
                if name.startswith('0') or name.startswith('2'):
                    param.requires_grad=False
        else:
            pass

    def param_copy(self, target_model, model):
        if model:
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(param.data)

    def param_copy_plus(self, target_model, model):
        model_dict_keys = model.state_dict().keys()
        for n, p in target_model.named_parameters():
            if n in model_dict_keys:
                p.data.copy_(model.state_dict()[n])

    def copy_params_from_super_layer(self, layer_idx):
        super_layer = self.super_layers_pool[layer_idx]
        nas_layer = self.nas_layers[layer_idx]
        connect_dict = self.connects[layer_idx]
        normal_cell_connect = connect_dict['normal']
        reduce_cell_connect = connect_dict['reduce']

        for super_cell, nas_cell in zip(super_layer, nas_layer):
            # copy preproc0 and preproc1
            self.param_copy_plus(nas_cell.preproc0, super_cell.preproc0)
            self.param_copy_plus(nas_cell.preproc1, super_cell.preproc1)

            if super_cell.reduction:
                cell_connect = reduce_cell_connect
            else:
                cell_connect = normal_cell_connect

            for i, (super_hidden, nas_hidden) in enumerate(zip(super_cell.dag, nas_cell.dag)):
                hidden_connect = cell_connect[i]
                # k = 2
                for j in range(len(hidden_connect)):
                    connect = hidden_connect[j]
                    super_edge = super_hidden[connect[0]]
                    super_op = super_edge._ops[connect[1]]
                    nas_edge = nas_hidden[j]
                    if isinstance(nas_edge, ops.Identity):
                        break
                    nas_op = nas_edge[0]
                    # copy params
                    self.param_copy_plus(nas_op, super_op)
                    # self.param_copy(super_op, nas_op)

    def copy_params_from_nas_layer(self, layer_idx):
        super_layer = self.super_layers_pool[layer_idx]
        nas_layer = self.nas_layers[layer_idx]
        connect_dict = self.connects[layer_idx]
        normal_cell_connect = connect_dict['normal']
        reduce_cell_connect = connect_dict['reduce']

        for super_cell, nas_cell in zip(super_layer, nas_layer):
            # copy preproc0 and preproc1
            self.param_copy_plus(super_cell.preproc0, nas_cell.preproc0)
            self.param_copy_plus(super_cell.preproc1, nas_cell.preproc1)

            if super_cell.reduction:
                cell_connect = reduce_cell_connect
            else:
                cell_connect = normal_cell_connect

            for i, (super_hidden, nas_hidden) in enumerate(zip(super_cell.dag, nas_cell.dag)):
                hidden_connect = cell_connect[i]
                # k = 2
                for j in range(len(hidden_connect)):
                    connect = hidden_connect[j]
                    super_edge = super_hidden[connect[0]]
                    super_op = super_edge._ops[connect[1]]
                    nas_edge = nas_hidden[j]
                    if isinstance(nas_edge, ops.Identity):
                        break
                    nas_op = nas_edge[0]
                    # copy params
                    self.param_copy_plus(super_op, nas_op)
                    # self.param_copy(super_op, nas_op)

    ######################## -------------------------- ########################
    ######################## Functions for layer search ########################
    ######################## -------------------------- ########################

    def add_super_layer(self, C_cur, C_p, C_pp, reduction_p=False, reduction_cur=False, cell_num=3, is_slim=False):
        cells = nn.ModuleList()
        # reduction_idx = (cell_num + 1) // 2 - 1
        # the first cell(block) is downsample
        # reduction_idx = 0
        if self.res_stem:
            reduction_idx = 0
        else:
            reduction_idx = cell_num - 1

        for i in range(cell_num):
            if i == reduction_idx and reduction_cur:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(self.n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, is_slim)
            reduction_p = reduction
            cells.append(cell)
            C_cur_out = C_cur * self.n_nodes
            C_pp, C_p = C_p, C_cur_out

        return cells

    def add_architecture_params(self, n_ops):
        arch_params = nn.ModuleList()

        alpha_normal = nn.ParameterList()
        alpha_reduce = nn.ParameterList()
        beta_normal = nn.ParameterList()
        beta_reduce = nn.ParameterList()

        for i in range(self.n_nodes):
            alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            if self.use_beta:
                beta_normal.append(nn.Parameter(1e-3*torch.randn(i+2)))
                beta_reduce.append(nn.Parameter(1e-3*torch.randn(i+2)))
            else:
                beta_normal.append(nn.Parameter(1e-1*torch.ones(i+2), requires_grad=False))
                beta_reduce.append(nn.Parameter(1e-1*torch.ones(i+2), requires_grad=False))

        arch_params.append(alpha_normal)
        arch_params.append(alpha_reduce)
        arch_params.append(beta_normal)
        arch_params.append(beta_reduce)

        return arch_params

    def pretrain_architecture_params(self, n_ops):
        arch_params = nn.ModuleList()

        alpha_normal = nn.ParameterList()
        alpha_reduce = nn.ParameterList()
        beta_normal = nn.ParameterList()
        beta_reduce = nn.ParameterList()

        for i in range(self.n_nodes):
            alpha_normal.append(nn.Parameter(1e-3*torch.ones(i+2, n_ops), requires_grad=False))
            alpha_reduce.append(nn.Parameter(1e-3*torch.ones(i+2, n_ops), requires_grad=False))
            beta_normal.append(nn.Parameter(1e-1*torch.ones(i+2), requires_grad=False))
            beta_reduce.append(nn.Parameter(1e-1*torch.ones(i+2), requires_grad=False))

        arch_params.append(alpha_normal)
        arch_params.append(alpha_reduce)
        arch_params.append(beta_normal)
        arch_params.append(beta_reduce)

        return arch_params

    ######################## ---------------------------- ########################
    ######################## Functions for layer generate ########################
    ######################## ---------------------------- ########################

    def generate_nas_layer(self, C_cur, C_p, C_pp, reduction_p, reduction_cur, genotype, cell_num=3, bn_affine=True):
        cells = nn.ModuleList()
        # reduction_idx = (cell_num + 1) // 2 - 1
        # the first cell(block) is downsample
        # reduction_idx = 0
        if self.res_stem:
            reduction_idx = 0
        else:
            reduction_idx = cell_num - 1

        for i in range(cell_num):
            if i == reduction_idx and reduction_cur:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction, bn_affine)
            reduction_p = reduction
            cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

        return cells

    ######################## ---------------------------- ########################
    ######################## Functions for stem           ########################
    ######################## ---------------------------- ########################
    def resnet_stem(self, inplanes=64):
        C_in = self.c_in
        feature_extractor = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(C_in, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            # the layer1 is concated with maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        feature_extractor.append(stem)
        return feature_extractor

    def cifar_stem(self, init_channel):
        C_in = self.c_in
        C_cur = init_channel
        feature_extractor = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        feature_extractor.append(stem)
        return feature_extractor
    
    def imagenet_stem(self, init_channel):
        C_in = self.c_in
        C_cur = init_channel
        feature_extractor = nn.ModuleList()
        stem0 = nn.Sequential(
            nn.Conv2d(C_in, C_cur // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_cur // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_cur // 2, C_cur, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_cur),
        )

        stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_cur, C_cur, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_cur),
        )
        feature_extractor.append(stem0)
        feature_extractor.append(stem1)
        return feature_extractor

    ######################## ---------------------------- ########################
    ######################## Functions for forward        ########################
    ######################## ---------------------------- ########################

    def extract_features(self, im):
        # feature_extractor is nn.ModuleList()
        if len(self.feature_extractor) == 1:
            s0 = self.feature_extractor[0](im)
            s1 = s0
            return [s0, s1]
        elif len(self.feature_extractor) == 2:
            s0 = self.feature_extractor[0](im)
            s1 = self.feature_extractor[1](s0)
            return [s0, s1]
        else:
            raise NotImplementedError

    def init_arch_params(self, layer_idx):
        init_arch_params = self.add_architecture_params(n_ops=len(ops.PRIMITIVES))
        for i in range(layer_idx, len(self.super_layers_arch)):
            target_arch = self.super_layers_arch[i]
            self.param_copy(target_arch, init_arch_params)

        for i in range(layer_idx, len(self.super_layers_pool_arch)):
            target_arch = self.super_layers_pool_arch[i]
            self.param_copy(target_arch, init_arch_params)

        del init_arch_params
    
    def freeze_arch_params(self, layer_idx=0):
        for i in range(self.super_layers_num):
            if i != layer_idx:
                for name, param in self.super_layers_arch[i].named_parameters():
                    param.requires_grad=False
            else:
                for name, param in self.super_layers_arch[i].named_parameters():
                    param.requires_grad=True

    def print_arch_params(self, logger, layer_idx=0):
        # remove formats
        if self.repeat_cell:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[0]
        else:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[layer_idx]
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        if self.use_beta:
            logger.info("####### BETA #######")
            logger.info("# Beta - normal")
            for beta in beta_normal:
                logger.info(F.softmax(beta, dim=-1))

            logger.info("\n# Beta - reduce")
            for beta in beta_reduce:
                logger.info(F.softmax(beta, dim=-1))
            logger.info("#####################")
        
    def generate_genotype(self, layer_idx=0):
        # arch_params list
        if self.repeat_cell:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[0]
        else:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[layer_idx]

        weights_normal = [F.softmax(alpha, dim=-1) for alpha in alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in alpha_reduce]
        weights_edge_normal = [F.softmax(beta, dim=0) for beta in beta_normal]
        weights_edge_reduce = [F.softmax(beta, dim=0) for beta in beta_reduce]

        gene_normal, connect_normal = gt.parse(weights_normal, weights_edge_normal, k=2)
        gene_reduce, connect_reduce = gt.parse(weights_reduce, weights_edge_reduce, k=2)
        connect_dict = {"normal": connect_normal, "reduce": connect_reduce}
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat), connect_dict

    def generate_genotype_gumbel(self, layer_idx=0):
        # arch_params list
        if self.repeat_cell:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[0]
        else:
            alpha_normal, alpha_reduce, beta_normal, beta_reduce = self.super_layers_arch[layer_idx]

        weights_normal = [F.softmax(alpha, dim=-1) for alpha in alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in alpha_reduce]
        weights_edge_normal = [F.softmax(beta, dim=0) for beta in beta_normal]
        weights_edge_reduce = [F.softmax(beta, dim=0) for beta in beta_reduce]

        gene_normal, connect_normal = gt.parse_gumbel(weights_normal, weights_edge_normal, k=2)
        gene_reduce, connect_reduce = gt.parse_gumbel(weights_reduce, weights_edge_reduce, k=2)
        connect_dict = {"normal": connect_normal, "reduce": connect_reduce}
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat), connect_dict

    def get_aux_logits(self, idx, s1):
        if idx == self.layer_num-3:
            return self.distill_aux_head1(s1)
        if idx == self.layer_num-2:
            return self.distill_aux_head2(s1)
        return None
        
    def forward(self, x,  layer_idx, super_flag=True, pretrain_flag=False, is_slim=False):
        # layer_idx, which stage we are
        # if super_flag, forward supernetwork else forward nas network
        # if pretrain_flag, foward supernetwork pool
        if pretrain_flag:
            super_layers_num = len(self.super_layers)
            nas_layers_num = 0
            super_layers = self.super_layers_pool
            super_layers_arch = self.super_layers_pool_arch
        else: 
            if super_flag:
                super_layers = self.super_layers
                super_layers_arch = self.super_layers_arch
                nas_layers = self.nas_layers
                nas_layers_num = len(self.nas_layers[:layer_idx])
                super_layers_num = len(self.super_layers[layer_idx:])
            else:
                nas_layers = self.nas_layers
                nas_layers_num = len(self.nas_layers)
                super_layers_num = 0
            
        outputs = []
        s0, s1 = self.extract_features(x)

        for i in range(nas_layers_num):
            s0, s1 = self.forward_nas_layer(s0, s1, nas_layers[i])
            logit = self.get_aux_logits(i, s1)
            if logit is not None:
                outputs.append(logit)

        aux_logits = None
        for j in range(super_layers_num):
            k = nas_layers_num + j
            if self.repeat_cell or pretrain_flag:
                s0, s1 = self.forward_super_layer(s0, s1, super_layers[k], super_layers_arch[0], is_slim)
                if k == self.layer_num-2:
                    aux_logits = self.distill_aux_head2(s1)
            else:
                s0, s1 = self.forward_super_layer(s0, s1, super_layers[k], super_layers_arch[k], is_slim)
            
            if not pretrain_flag:
                logit = self.get_aux_logits(k, s1)
                if logit is not None:
                    outputs.append(logit)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        if super_flag:
            logits = self.fc_super(out)
        else:
            logits = self.fc_nas(out)
        
        if pretrain_flag:
            return logits, aux_logits

        outputs.append(logits)
        logits_output = logits

        ensemble_param = F.softmax(self.ensemble_param, dim=0)
        if self.ensemble_sum:
            em_output = ensemble_param[0] * outputs[0] + ensemble_param[1] * outputs[1] + ensemble_param[2] * outputs[2]
        else:
            em_output = torch.cat((ensemble_param[0] * outputs[0], ensemble_param[1] * outputs[1], ensemble_param[2] * outputs[2]), 0)

        return logits_output, em_output
        # return em_output, em_output

    def process_alpha(self, alpha_param, beta_param):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in alpha_param]
        weights_edge_normal = [F.softmax(beta, dim=0) for beta in beta_param]
        output_alpha = nn.ParameterList()
        for alpha in weights_normal:
            output_alpha.append(nn.Parameter(torch.zeros_like(alpha), requires_grad=False))

        connect_idx = []
        k = 2
        for idx, (edges, w) in enumerate(zip(weights_normal, weights_edge_normal)):
            # edges: Tensor(n_edges, n_ops)
            edge_max, primitive_indices = torch.topk((w.view(-1, 1) * edges)[:, :-1], 1) # ignore 'none'
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_idx = []
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                node_idx.append((edge_idx.item(), prim_idx.item()))
                output_alpha[idx][edge_idx.item(), prim_idx.item()] = 1.

            connect_idx.append(node_idx)

        return output_alpha

    def forward_super_layer(self, s0, s1, super_layer, arch_params, is_slim=False):
        # arch_params: list
        # super_layer: cells (2 / 3)

        alpha_normal, alpha_reduce, beta_normal, beta_reduce = arch_params
        if is_slim:
            weights_normal = self.process_alpha(alpha_normal, beta_normal)
            weights_edge_normal = [F.softmax(beta, dim=0) for beta in beta_normal]
            weights_reduce = self.process_alpha(alpha_reduce, beta_reduce)
            weights_edge_reduce = [F.softmax(beta, dim=0) for beta in beta_reduce]
        else:
            weights_normal = [F.softmax(alpha, dim=-1) for alpha in alpha_normal]
            weights_edge_normal = [F.softmax(beta, dim=0) for beta in beta_normal]
            weights_reduce = [F.softmax(alpha, dim=-1) for alpha in alpha_reduce]
            weights_edge_reduce = [F.softmax(beta, dim=0) for beta in beta_reduce]

        for cell in super_layer:
            weights = weights_reduce if cell.reduction else weights_normal
            weights_edge = weights_edge_reduce if cell.reduction else weights_edge_normal
            s0, s1 = s1, cell(s0, s1, weights, weights_edge)

        return s0, s1

    def forward_nas_layer(self, s0, s1, nas_layer):
        
        for cell in nas_layer:
            s0, s1 = s1, cell(s0, s1)
        
        return s0, s1

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def add_alpha_regularization(self, operations, weight_decay=0.0005, method='L2', normal=True, reduce=True):
        if method == 'L2':
            reg_loss = torch.tensor(0.).to(torch.device("cuda"))
            for operation in operations:
                if self.repeat_cell:
                    stage, operation = operation
                    stage = 0
                else:
                    stage, operation = operation
                if normal:
                    for node in self.super_layers_arch[stage][0]:
                        for connection in node:
                            reg_loss += connection[ops.PRIMITIVES.index(operation)] * \
                                        connection[ops.PRIMITIVES.index(operation)]
                if reduce:
                    for node in self.super_layers_arch[stage][1]:
                        for connection in node:
                            reg_loss += connection[ops.PRIMITIVES.index(operation)] * \
                                        connection[ops.PRIMITIVES.index(operation)]
            return reg_loss * weight_decay
        elif method == 'L1':
            reg_loss = torch.tensor(0.).cuda()
            for operation in operations:
                if self.repeat_cell:
                    stage, operation = operation
                    stage = 0
                else:
                    stage, operation = operation
                    
                if normal:
                    for node in self.super_layers_arch[stage][0]:
                        for connection in node:
                            reg_loss += abs(connection[ops.PRIMITIVES.index(operation)])
                if reduce:
                    for node in self.super_layers_arch[stage][1]:
                        for connection in node:
                            reg_loss += abs(connection[ops.PRIMITIVES.index(operation)])
            return reg_loss * weight_decay
        else:
            raise ValueError('Method isn\'t supported')