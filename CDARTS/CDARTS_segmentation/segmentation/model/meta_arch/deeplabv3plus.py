# ------------------------------------------------------------------------------
# DeepLabV3+ meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn

from .base import BaseSegmentationModel
from segmentation.model.decoder import DeepLabV3PlusDecoder
from segmentation.utils import AverageMeter


__all__ = ["DeepLabV3Plus"]


class DeepLabV3Plus(BaseSegmentationModel):
    """
    Implements DeepLabV3+ model from
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        in_channels (int): number of input channels from the backbone
        feature_key (str): name of input feature from backbone
        low_level_channels (int): channels of low-level features
        low_level_key (str): name of low-level features used in decoder
        low_level_channels_project (int): channels of low-level features after projection in decoder
        decoder_channels (int): number of channels in decoder
        atrous_rates (tuple): atrous rates for ASPP
        num_classes (int): number of classes
        semantic_loss (nn.Module): loss function
        semantic_loss_weight (float): loss weight
    """

    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key,
                 low_level_channels_project, decoder_channels, atrous_rates, num_classes,
                 semantic_loss, semantic_loss_weight, **kwargs):
        decoder = DeepLabV3PlusDecoder(in_channels, feature_key, low_level_channels, low_level_key,
                                       low_level_channels_project, decoder_channels, atrous_rates, num_classes)
        super(DeepLabV3Plus, self).__init__(backbone, decoder)

        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight

        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()

        # Initialize parameters.
        self._init_params()

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        if targets is not None:
            semantic_loss = self.semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            results['loss'] = semantic_loss
        return results
