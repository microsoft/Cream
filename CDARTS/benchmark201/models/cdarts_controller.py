import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy

from models.search_cells import SearchCell
from models.augment_cells import InferCell
from models.aux_head import DistillHeadCIFAR
from models.ops import ResNetBasicblock, OPS, NAS_BENCH_201
from utils.genotypes import Structure

class CDARTSController(nn.Module):
    """ CDARTS Controller"""
    def __init__(self, config, criterion, n_nodes=4, stem_multiplier=3, track_running_stats=True):
        """
        args:

        """
        super(CDARTSController, self).__init__()

        # some settings
        self.n_nodes = n_nodes
        self.criterion = criterion
        self.layer_num = config.layer_num
        self.c_in = config.input_channels
        self.num_classes = config.n_classes
        # cifar10 or imagenet
        self.model_type = config.model_type
        self.stem_multiplier = stem_multiplier
        self.init_channel = config.init_channels
        self.ensemble_sum = config.ensemble_sum
        self.use_ensemble_param = config.ensemble_param
        self.bn_affine = config.bn_affine
        self.fix_head = config.fix_head
        self.share_fc = config.share_fc

        self.layers = [6, 6, 5]
        self.layers_reduction = [True, True, False]
        self.augment_layers = [6, 6, 5]
        self.num_edge = None
        self.edge2index = None
        self.nas_genotype = None
        self.cell_connects = {}
        self.search_space = NAS_BENCH_201
        self.op_names = copy.deepcopy(self.search_space)
        self.track_running_stats = track_running_stats
        
        self.fc_super = None
        self.fc_nas = None
        self.distill_aux_c1 = None
        self.distill_aux_c2 = None
        self.feature_extractor = None
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.nas_layers = nn.ModuleList([None, None, None])
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
        else:
            raise Exception("error! not support now!")

        c_p = self.init_channel * self.stem_multiplier
        c_cur = self.init_channel


        for layer_idx in range(self.layer_num):
            reduction = self.layers_reduction[layer_idx]
            super_layer = self.add_super_layer(c_cur, c_p, reduction, self.layers[layer_idx]) 
            super_layer_pool = self.add_super_layer(c_cur, c_p, reduction, self.augment_layers[layer_idx])

            self.super_layers.append(super_layer)
            self.super_layers_pool.append(super_layer_pool)

            if reduction:
                c_cur = c_cur * 2
            else:
                c_cur = c_cur
            c_p = c_cur

            if layer_idx == self.layer_num-3:
                self.distill_aux_c1 = c_p
            if layer_idx == self.layer_num-2:
                self.distill_aux_c2 = c_p

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
            self.distill_aux_head1 = DistillHeadCIFAR(self.distill_aux_c1, 6, self.num_classes, bn_affine=self.bn_affine)
            self.distill_aux_head2 = DistillHeadCIFAR(self.distill_aux_c2, 6, self.num_classes, bn_affine=self.bn_affine)
        else:
            raise Exception("error! not support now!")

        self._arch_parameters = nn.Parameter( 1e-3*torch.randn(self.num_edge, len(self.search_space)) )
        self.fix_structure()

    def fix_structure(self):
        if self.fix_head:
            for n, p in self.distill_aux_head1.named_parameters():
                p.requires_grad = False
            for n, p in self.distill_aux_head2.named_parameters():
                p.requires_grad = False

    def build_nas_model(self, genotype):
        c_p = self.init_channel * self.stem_multiplier
        c_cur = self.init_channel

        for i in range(self.layer_num):
            reduction = self.layers_reduction[i]
                
            self.nas_layers[i]  = self.add_nas_layer(c_cur, c_p, reduction, genotype, self.augment_layers[i])

            if reduction:
                c_cur = c_cur * 2
            else:
                c_cur = c_cur
            c_p = c_cur
 
    def param_copy_plus(self, target_model, model):
        if model:
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(param.data)

    def param_copy_plus1(self, target_model, model):
        model_dict_keys = model.state_dict().keys()
        for n, p in target_model.named_parameters():
            if n in model_dict_keys:
                p.data.copy_(model.state_dict()[n])

    def copy_params_from_super_layer(self):
        for layer_idx in range(self.layer_num):
            super_layer = self.super_layers_pool[layer_idx]
            nas_layer = self.nas_layers[layer_idx]
            for super_cell, nas_cell in zip(super_layer, nas_layer):
                if isinstance(super_cell, ResNetBasicblock) and isinstance(nas_cell, ResNetBasicblock):
                    self.param_copy_plus(nas_cell, super_cell)
                else:
                    for edge_key, nas_op in zip(super_cell._modules['edges'].keys(), nas_cell._modules['layers']):
                        self.param_copy_plus(nas_op, super_cell._modules['edges'][edge_key][self.cell_connects[edge_key]])
                                    
    def copy_params_from_nas_layer(self):
        for layer_idx in range(self.layer_num):
            super_layer = self.super_layers_pool[layer_idx]
            nas_layer = self.nas_layers[layer_idx]
            for super_cell, nas_cell in zip(super_layer, nas_layer):
                if isinstance(super_cell, ResNetBasicblock) and isinstance(nas_cell, ResNetBasicblock):
                    self.param_copy_plus(super_cell, nas_cell)
                else:
                    for edge_key, nas_op in zip(super_cell._modules['edges'].keys(), nas_cell._modules['layers']):
                        self.param_copy_plus(super_cell._modules['edges'][edge_key][self.cell_connects[edge_key]], nas_op)
    
    ######################## -------------------------- ########################
    ######################## Functions for layer search ########################
    ######################## -------------------------- ########################

    def add_super_layer(self, C_cur, C_p, reduction_cur=False, cell_num=3):
        cells = nn.ModuleList()
        reduction_idx = cell_num - 1

        for i in range(cell_num):
            if i == reduction_idx and reduction_cur:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            
            if reduction:
                cell = ResNetBasicblock(C_p, C_cur, 2)
            else:
                cell = SearchCell(C_p, C_cur, 1, self.n_nodes, self.search_space, self.bn_affine, self.track_running_stats)
                if self.num_edge is None: self.num_edge, self.edge2index = cell.num_edges, cell.edge2index
                else: assert self.num_edge == cell.num_edges and self.edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(self.num_edge, cell.num_edges)

            cells.append(cell)
            C_p = cell.out_dim

        return cells

    ######################## ---------------------------- ########################
    ######################## Functions for layer generate ########################
    ######################## ---------------------------- ########################

    def add_nas_layer(self, C_cur, C_p, reduction_cur, genotype, cell_num=3):
        cells = nn.ModuleList()
        reduction_idx = cell_num - 1

        for i in range(cell_num):
            if i == reduction_idx and reduction_cur:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            if reduction:
                cell = ResNetBasicblock(C_p, C_cur, 2, True)
            else:
                cell = InferCell(genotype, C_p, C_cur, 1)

            cells.append(cell)
            C_p = cell.out_dim

        return cells

    ######################## ---------------------------- ########################
    ######################## Functions for stem           ########################
    ######################## ---------------------------- ########################

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
        
    def get_aux_logits(self, idx, s1):
        if idx == self.layer_num-3:
            return self.distill_aux_head1(s1)
        if idx == self.layer_num-2:
            return self.distill_aux_head2(s1)
        return None
        
    def forward(self, x, super_flag=True, updateType='alpha'):

        if super_flag:
            super_layers = self.super_layers
            nas_layers_num = 0
            super_layers_num = len(self.super_layers)
        else:
            nas_layers = self.nas_layers
            nas_layers_num = len(self.nas_layers)
            super_layers_num = 0
            
        outputs = []
        s0, s1 = self.extract_features(x)

        for i in range(nas_layers_num):
            s1 = self.forward_nas_layer(s1, nas_layers[i])
            logit = self.get_aux_logits(i, s1)
            if logit is not None:
                outputs.append(logit)

        for j in range(super_layers_num):
            k = nas_layers_num + j
            s1 = self.forward_super_layer(s1, super_layers[k], updateType)
            logit = self.get_aux_logits(k, s1)
            if logit is not None:
                outputs.append(logit)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        if super_flag:
            logits = self.fc_super(out)
        else:
            logits = self.fc_nas(out)
        
        outputs.append(logits)
        logits_output = logits

        ensemble_param = F.softmax(self.ensemble_param, dim=0)
        if self.ensemble_sum:
            em_output = ensemble_param[0] * outputs[0] + ensemble_param[1] * outputs[1] + ensemble_param[2] * outputs[2]
        else:
            em_output = torch.cat((ensemble_param[0] * outputs[0], ensemble_param[1] * outputs[1], ensemble_param[2] * outputs[2]), 0)

        return logits_output, em_output

    def forward_super_layer(self, s1, super_layer, updateType='alpha'):
        if updateType == 'weight':
            alphas = self._arch_parameters
        else:
            alphas  = F.softmax(self._arch_parameters, dim=-1)

        for cell in super_layer:
            if isinstance(cell, SearchCell):
                s1 = cell(s1, alphas)
            else:
                s1 = cell(s1)
        return s1

    def forward_nas_layer(self, s1, nas_layer):
        for cell in nas_layer:
            s1 = cell(s1)
        return s1

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def genotype(self):
        genotypes = []
        for i in range(1, self.n_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self._arch_parameters[ self.edge2index[node_str] ]
                    op_name = self.op_names[ weights.argmax().item() ]
                    self.cell_connects[node_str] = weights.argmax().item()
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        self.nas_genotype = Structure(genotypes)
        return self.nas_genotype

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self._arch_parameters, dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def _save_arch_parameters(self):
        self._saved_arch_parameters = self._arch_parameters.clone()

    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        self._arch_parameters.data.copy_(F.softmax(self._arch_parameters, dim=-1))

    def restore_arch_parameters(self):
        self._arch_parameters.data.copy_(self._saved_arch_parameters)
        del self._saved_arch_parameters

    def arch_parameters(self):
        return [self._arch_parameters]

    def l1_loss(self):
        return torch.mean(torch.abs(self._arch_parameters[:, 0:1]))


