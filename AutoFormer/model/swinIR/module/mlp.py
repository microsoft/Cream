import torch.nn as nn
import torch.nn.functional as F
from AutoFormer.model.vision_transformer.module.Linear_super import LinearSuper

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearSuper(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = LinearSuper(hidden_features, out_features)
        # Note: We'll actually only end up using `sample_drop`.
        # `super_drop` is only for recording what was originally passed.
        self.super_drop = drop
        self.sample_drop = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.sample_drop, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_drop, training=self.training)
        return x

    def set_sample_config(self,
                          sample_embed_dim=None,
                          sample_mlp_ratio=None,
                          sample_drop=None,
                          ):
        self.fc1.set_sample_config(int(sample_embed_dim), int(sample_embed_dim * sample_mlp_ratio))
        self.fc2.set_sample_config(int(sample_embed_dim * sample_mlp_ratio), int(sample_embed_dim))
        self.sample_drop = sample_drop

    #Return 0, Linear layer return params
    def calc_sampled_param_num(self):
        return 0
