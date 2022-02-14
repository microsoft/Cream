""" CNN cell for network augmentation """
import torch.nn as nn
from copy import deepcopy
from models.ops import OPS


# Cell for NAS-Bench-201
class InferCell(nn.Module):

  def __init__(self, genotype, C_in, C_out, stride):
    super(InferCell, self).__init__()

    self.layers  = nn.ModuleList()
    self.node_IN = []
    self.node_IX = []
    self.genotype = deepcopy(genotype)
    for i in range(1, len(genotype)):
      node_info = genotype[i-1]
      cur_index = []
      cur_innod = []
      for (op_name, op_in) in node_info:
        if op_in == 0:
          layer = OPS[op_name](C_in , C_out, stride, True, True)
        else:
          layer = OPS[op_name](C_out, C_out,      1, True, True)
        cur_index.append( len(self.layers) )
        cur_innod.append( op_in )
        self.layers.append( layer )
      self.node_IX.append( cur_index )
      self.node_IN.append( cur_innod )
    self.nodes   = len(genotype)
    self.in_dim  = C_in
    self.out_dim = C_out

  def extra_repr(self):
    string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    laystr = []
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
      x = '{:}<-({:})'.format(i+1, ','.join(y))
      laystr.append( x )
    return string + ', [{:}]'.format( ' | '.join(laystr) ) + ', {:}'.format(self.genotype.tostr())

  def forward(self, inputs):
    nodes = [inputs]
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      node_feature = sum( self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods) )
      nodes.append( node_feature )
    return nodes[-1]
