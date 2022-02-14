import torch
import torch.nn as nn
from models import ops
from models.augment_cells import AugmentCell

class ModelTest(nn.Module):

    def __init__(self, genotypes_dict, model_type, res_stem=False, init_channel=96, stem_multiplier=3, n_nodes=4, num_classes=1000):
        """
        args:

        """
        super(ModelTest, self).__init__()
        self.c_in = 3
        self.init_channel = init_channel
        self.stem_multiplier = stem_multiplier
        self.num_classes = num_classes
        self.n_nodes = n_nodes
        self.model_type = model_type
        self.res_stem = res_stem

        if self.model_type == 'cifar':
            reduction_p = False
            self.layers_reduction = [True, True, False]
            self.augment_layers = [7, 7, 6]
            self.nas_layers = nn.ModuleList([None, None, None])
            self.feature_extractor = self.cifar_stem(self.init_channel * self.stem_multiplier)

        elif self.model_type == 'imagenet':
            if self.res_stem:
                reduction_p = False
                self.nas_layers = nn.ModuleList([None, None, None, None])
                self.layers_reduction = [False, True, True, True]
                self.augment_layers = [3, 4, 3, 4]
                self.feature_extractor = self.resnet_stem(self.init_channel * self.stem_multiplier)
            else:
                reduction_p = True
                self.nas_layers = nn.ModuleList([None, None, None])
                self.layers_reduction = [True, True, False]
                self.augment_layers = [5, 5, 4]
                self.feature_extractor = self.imagenet_stem(self.init_channel * self.stem_multiplier)
        else:
            raise Exception("Wrong model type!")

        self.nas_layers_num = len(self.nas_layers)
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

        self.fc = nn.Linear(c_p, self.num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def generate_nas_layer(self, C_cur, C_p, C_pp, reduction_p, reduction_cur, genotype, cell_num=3, bn_affine=True):
        cells = nn.ModuleList()
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

    def forward(self, x):
        s0, s1 = self.extract_features(x)
        for i in range(self.nas_layers_num):
            s0, s1 = self.forward_nas_layer(s0, s1, self.nas_layers[i])

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.fc(out)
        return logits, logits

    def forward_nas_layer(self, s0, s1, nas_layer):
        for cell in nas_layer:
            s0, s1 = s1, cell(s0, s1)
        return s0, s1

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

