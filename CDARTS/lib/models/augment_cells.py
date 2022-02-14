""" CNN cell for network augmentation """
import torch
import torch.nn as nn
from lib.models import ops
import lib.utils.genotypes as gt


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction, bn_affine=True):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=bn_affine)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=bn_affine)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=bn_affine)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = gt.to_dag(C, gene, reduction, bn_affine)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out
