# ------------------------------------------------------------------------------
# Loss functions.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F


class RegularCE(nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        pixel_losses = pixel_losses[mask]
        return pixel_losses.mean()


class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, threshold=0.7,
                 min_kept=100000, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        predictions = F.softmax(logits, dim=1)
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        # Get the score for gt class at each pixel location.
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = predictions.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)

        pixel_losses = pixel_losses[mask][indices]
        pixel_losses = pixel_losses[predictions < threshold]
        return pixel_losses.mean()


class DeepLabCE(nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()
