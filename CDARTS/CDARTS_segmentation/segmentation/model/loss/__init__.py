from torch import nn

from .criterion import RegularCE, OhemCE, DeepLabCE

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss
