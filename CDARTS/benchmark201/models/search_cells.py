""" CNN cell for architecture search """
import torch
import torch.nn as nn
from copy import deepcopy
from models.ops import ResNetBasicblock, OPS, NAS_BENCH_201
from utils.genotypes import Structure

# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
    super(SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in , C_out, stride, affine, track_running_stats) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in , C_out,      1, affine, track_running_stats) for op_name in op_names]
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # GDAS
  def forward_gdas(self, inputs, hardwts, index):
    nodes   = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = hardwts[ self.edge2index[node_str] ]
        argmaxs  = index[ self.edge2index[node_str] ].item()
        weigsum  = sum( weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie] for _ie, edge in enumerate(self.edges[node_str]) )
        inter_nodes.append( weigsum )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # joint
  def forward_joint(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        #aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
        aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) )
        inter_nodes.append( aggregation )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # uniform random sampling per iteration, SETN
  def forward_urs(self, inputs):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      while True: # to avoid select zero for all ops
        sops, has_non_zero = [], False
        for j in range(i):
          node_str   = '{:}<-{:}'.format(i, j)
          candidates = self.edges[node_str]
          select_op  = random.choice(candidates)
          sops.append( select_op )
          if not hasattr(select_op, 'is_zero') or select_op.is_zero is False: has_non_zero=True
        if has_non_zero: break
      inter_nodes = []
      for j, select_op in enumerate(sops):
        inter_nodes.append( select_op(nodes[j]) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_select(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str][ weights.argmax().item() ]( nodes[j] ) )
        #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # forward with a specific structure
  def forward_dynamic(self, inputs, structure):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      cur_op_node = structure.nodes[i-1]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( nodes[j] ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]
