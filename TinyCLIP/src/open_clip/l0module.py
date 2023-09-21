# Adapted from https://github.com/princeton-nlp/CoFiPruning/blob/main/models/l0_module.py
# MIT license
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class L0Module(nn.Module):
    limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
    all_types = ["hidden_z", "heads_z", "mha_z", "intermediate_z", "ffn_z"]

    def __init__(self, config,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 lagrangian_warmup=0,
                 init_loga=0.5,
                 temperature=2. / 3.,
                 pruning_type=["hidden", "heads", "intermediate", "layer"],
                 magical_number=0.8,  # from Wang et al. 2020
                 ):
        super(L0Module, self).__init__()

        self.magical_number = magical_number
        self.lagrangian_warmup = lagrangian_warmup

        self.pruning_type = pruning_type
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity
        self.temperature = temperature

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.dim_per_head = self.hidden_size // self.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers

        self.params_per_head_layer = self.hidden_size * \
            self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head = self.params_per_head_layer // self.num_attention_heads

        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * \
            2 + self.hidden_size + self.intermediate_size
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (
            self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0

        init_loga = init_loga if isinstance(init_loga, float) else 0.5
        self.loga_mean = math.log(
            1.0 - self.epsilon - init_loga) - math.log(init_loga + self.epsilon)

        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        for t in pruning_type:
            self.initialize_one_module(t)

        self.lambda_1 = nn.Parameter(torch.tensor(10.00))
        self.lambda_2 = nn.Parameter(torch.tensor(10.00))

    def initialize_parameters(self, size, num_layer=None, mean=None):
        if num_layer is not None:
            loga = nn.Parameter(torch.Tensor(num_layer, size))
        else:
            loga = nn.Parameter(torch.Tensor(size))
        mean = mean or self.loga_mean
        # loga.data.normal_(mean, 1e-2)
        loga.data.normal_(mean, 0)
        return loga

    def initialize_one_module(self, module_name):
        default_mean = 10
        if module_name == "intermediate":
            self.intermediate_loga = self.initialize_parameters(
                self.intermediate_size, self.num_hidden_layers, mean=default_mean)
            self.add_one_module(
                self.intermediate_loga, type_name="intermediate",
                parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                shape=[self.num_hidden_layers, 1, 1, self.intermediate_size]
            )
            self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        elif module_name == "heads":
            self.heads_loga = self.initialize_parameters(
                self.num_attention_heads, self.num_hidden_layers, mean=default_mean)
            self.add_one_module(
                self.heads_loga, type_name="heads",
                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                shape=[self.num_hidden_layers, 1,
                       self.num_attention_heads, 1, 1]
            )
            self.prunable_model_size += self.params_per_head * \
                self.num_hidden_layers * self.num_attention_heads
        elif module_name == "hidden":
            self.hidden_loga = self.initialize_parameters(
                self.hidden_size, mean=default_mean)
            self.add_one_module(
                self.hidden_loga, type_name="hidden",
                parameter_per_dim=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                size=self.hidden_size, shape=[self.hidden_size]
            )
        elif module_name == "layer":
            self.ffn_loga = self.initialize_parameters(
                self.num_hidden_layers, mean=default_mean)
            self.add_one_module(
                self.ffn_loga, type_name="ffn",
                parameter_per_dim=self.params_per_mlp_layer, size=1,
                shape=[self.num_hidden_layers]
            )
            self.mha_loga = self.initialize_parameters(
                self.num_hidden_layers, mean=default_mean)
            self.add_one_module(
                self.mha_loga, type_name="mha",
                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                shape=[self.num_hidden_layers]
            )

    # ! init the z_logas
    def add_one_module(self, z_loga, type_name, parameter_per_dim, size, shape):
        self.types.append(type_name)
        self.z_logas[type_name] = z_loga
        self.parameters_per_dim[type_name] = parameter_per_dim
        self.sizes[type_name] = size
        self.shapes[type_name] = shape

    def constrain_parameters(self):
        for key in self.z_logas:
            self.z_logas[key].data.clamp_(
                min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1.0 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def score_loga(self, loga):
        return 1.0 - self.cdf_qz(0.0, loga)

    def get_num_parameters_and_constraint(self, hidden=False):
        num_parameters = 0

        layers = self.num_hidden_layers
        hidden_size = self.hidden_size
        heads = self.num_attention_heads
        device = self.z_logas[self.types[0]].device
        # 12 * 1 * 1
        mha_score = self.score_loga(self.mha_loga).view(
            -1, 1, 1) if "mha" in self.types else torch.ones([layers, 1, 1]).to(device)
        # 12 * 12 * 1
        heads_score = self.score_loga(self.heads_loga).unsqueeze(
            dim=-1) if "heads" in self.types else torch.ones([layers, heads, 1]).to(device)

        if "heads" not in self.parameters_per_dim:
            self.parameters_per_dim["heads"] = self.params_per_head
        if "intermediate" not in self.parameters_per_dim:
            self.parameters_per_dim["intermediate"] = self.params_per_intermediate_dim

        if hidden:
            hidden_score = self.score_loga(
                self.hidden_loga) if "hidden" in self.types else torch.ones([hidden_size]).to(device)
            heads_score = (
                heads_score * mha_score) if mha_score is not None else heads_score  # 38+106
            heads_score = heads_score.reshape(-1)
            num_parameters += torch.outer(hidden_score, heads_score).sum(
            ) * self.parameters_per_dim["heads"] / self.hidden_size
        else:
            heads_score = heads_score * mha_score
            num_parameters += heads_score.sum() * \
                self.parameters_per_dim["heads"]

        # 12 * 1
        if 'ffn' in self.types:
            ffn_score = self.score_loga(self.ffn_loga).unsqueeze(
                dim=-1) if "ffn" in self.types else torch.ones([layers, 1]).to(device)
        else:
            ffn_score = 1
        # 12 * 3072
        intermediate_score = self.score_loga(self.intermediate_loga) if "intermediate" in self.types else torch.ones([
            layers, hidden_size * 4]).to(device)

        intermediate_score = intermediate_score * ffn_score

        if hidden:
            intermediate_score = intermediate_score.reshape(-1)  # 13893+22971
            num_parameters += torch.sum(torch.outer(hidden_score,
                                        intermediate_score)) * 2
        else:
            num_parameters += intermediate_score.sum() * \
                self.parameters_per_dim["intermediate"]

        return num_parameters

    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * \
            min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity

    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.get_target_sparsity(
            pruned_steps) if self.lagrangian_warmup > 0 else self.target_sparsity
        expect_sparsity = 1 - self.get_num_parameters_and_constraint(
            "hidden" in self.types) / self.prunable_model_size

        # lagrangian_loss = (
        #     self.lambda_1 * (expect_sparsity - target_sparsity).abs() +
        #     self.lambda_2 * (expect_sparsity - target_sparsity).square()
        # )

        zero = torch.tensor(0.0, device=expect_sparsity.device)
        lagrangian_loss = (
            self.lambda_1 * torch.maximum(target_sparsity - expect_sparsity, zero) +
            self.lambda_2 *
            torch.maximum(target_sparsity - expect_sparsity, zero).square()
        )

        return lagrangian_loss, expect_sparsity.detach().item(), target_sparsity

    # during training
    def _sample_z(self, loga):
        # Uniform random numbers for the concrete distribution
        u = torch.zeros_like(loga).uniform_(self.epsilon, 1.0 - self.epsilon)
        # quantile concrete
        z = torch.sigmoid(
            (torch.log(u) - torch.log(1 - u) + loga) / self.temperature)
        z = z * (self.limit_b - self.limit_a) + self.limit_a
        z = F.hardtanh(z, min_val=0.0, max_val=1.0)
        return z

    # during inference
    def _deterministic_z(self, size, loga, soft=True):
        soft_mask = torch.sigmoid(
            loga / self.temperature * self.magical_number)
        if not soft:
            return soft_mask
        expected_num_zeros = size - self.score_loga(loga).sum().item()
        num_zeros = round(expected_num_zeros)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {}
        # for t in self.all_types:
        #     name = t[:-2]
        for t in self.types:
            name = t
            numpified_zs[name] = (zs[t].squeeze().detach().cpu(
            ).numpy() > 0) if t in zs else np.ones(self.shapes[name])
        return numpified_zs

    def calculate_model_size(self, zs):
        if zs is None:
            return {"pruned_sparsity": 0.0}

        layers = self.num_hidden_layers
        hidden_size = self.hidden_size
        heads = self.num_attention_heads
        device = self.z_logas[self.types[0]].device

        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs["hidden"] if "hidden" in numpified_zs.keys() else np.ones([
            hidden_size])
        heads_z = numpified_zs["heads"] if "heads" in numpified_zs.keys() else np.ones([
            layers, 1, heads, 1, 1])
        mha_z = numpified_zs["mha"].reshape(-1, 1, 1, 1, 1) if "mha" in numpified_zs.keys(
        ) else np.ones([heads_z.shape[0], 1, 1, 1, 1])
        intermediate_z = numpified_zs["intermediate"] if "intermediate" in numpified_zs.keys(
        ) else np.ones([layers, 1, 1, hidden_size * 4])
        ffn_z = numpified_zs["ffn"].reshape(-1, 1, 1, 1) if "ffn" in numpified_zs.keys(
        ) else np.ones([heads_z.shape[0], 1, 1, 1])

        remain_hidden = hidden_z.sum().item()
        remain_intermediate = intermediate_z.reshape(
            self.num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remain_heads = heads_z.reshape(
            self.num_hidden_layers, self.num_attention_heads).sum(-1).tolist()

        heads = np.outer((heads_z * mha_z).reshape(-1), hidden_z).sum().item()
        intermediate = np.outer(
            (intermediate_z * ffn_z).reshape(-1), hidden_z).sum().item()

        remain_model_size = heads * self.dim_per_head * 4 + intermediate * 2

        pruned_model_size = self.prunable_model_size - remain_model_size

        results = {
            'mha': mha_z.reshape(-1).astype(int).tolist(),
            'ffn': ffn_z.reshape(-1).astype(int).tolist(),
            'remain_hidden': remain_hidden,
            'remain_intermediate': remain_intermediate,
            'remain_heads': remain_heads,
            'pruned_params': pruned_model_size,
            'remain_params': remain_model_size,
            'pruned_sparsity': pruned_model_size / self.prunable_model_size
        }
        return results

    def forward(self, soft=True):
        zs = {f"{t}_z": [] for t in self.types}

        if self.training:
            for i, t in enumerate(self.types):
                loga = self.z_logas[t]
                z = self._sample_z(loga)
                zs[f"{t}_z"] = z.reshape(self.shapes[t])
        else:
            for i, t in enumerate(self.types):
                if t != "hidden":  # hidden is not a per layer sample
                    tmp = []
                    for loga in self.z_logas[t]:
                        z = self._deterministic_z(
                            self.sizes[t], loga.detach(), soft=soft)
                        tmp.append(z.reshape(self.shapes[t][1:]))
                    zs[f"{t}_z"] = torch.stack(tmp)
                else:
                    zs[f"{t}_z"] = self._deterministic_z(
                        self.sizes[t], self.hidden_loga.detach(), soft=soft)
        return zs

    @torch.no_grad()
    def l0_mask(self):
        zs = {f"{t}_z": [] for t in self.types}
        # self.magical_number = 1.0

        def get_mask(loga): return torch.sigmoid(
            loga / self.temperature * self.magical_number)
        for t in self.types:
            if t == "hidden":
                zs[f"{t}_z"] = get_mask(self.hidden_loga)
            else:
                tmp = []
                loga_all_layers = self.z_logas[t]
                for layer in range(len(loga_all_layers)):
                    loga = loga_all_layers[layer]
                    z = get_mask(loga)
                    tmp.append(z.reshape(self.shapes[t][1:]))
                zs[f"{t}_z"] = torch.stack(tmp)
        return zs


if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple('Config', [
                        'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_hidden_layers'])
    config = Config(hidden_size=768, intermediate_size=4 * 768,
                    num_attention_heads=12, num_hidden_layers=12)
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
    l0_module.train()
    zs = l0_module()
    l0_module.eval()
    zs = l0_module()
    result = l0_module.calculate_model_size(zs)
    print(result)
