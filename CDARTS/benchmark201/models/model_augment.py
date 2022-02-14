import torch
import torch.nn as nn

class ModelAug(nn.Module):

    def __init__(self, feature_extractor, nas_layers, fc_layer, n_nodes=4, aux_head=None):
        """
        args:

        """
        super(ModelAug, self).__init__()
        self.feature_extractor = feature_extractor

        self.nas_layers = nas_layers
        self.nas_layers_num = len(nas_layers)
        self.fc = fc_layer
        self.aux_head = aux_head
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        logits_aux = None
        if len(self.feature_extractor) == 1:
            s0 = self.feature_extractor[0](x)
            s1 = s0
        elif len(self.feature_extractor) == 2:
            s0 = self.feature_extractor[0](x)
            s1 = self.feature_extractor[1](s0)
        else:
            raise NotImplementedError

        for i in range(self.nas_layers_num):
            s1 = self.forward_nas_layer(s1, self.nas_layers[i])
            # if i == (self.nas_layers_num * 2 // 3 - 1):
            if i == (self.nas_layers_num - 2):
                if self.training:
                    logits_aux = self.aux_head(s1)
                    
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.fc(out)
        return logits, logits_aux

    def forward_nas_layer(self, s1, nas_layer):
        
        for cell in nas_layer:
            s1 = cell(s1)
        return s1

