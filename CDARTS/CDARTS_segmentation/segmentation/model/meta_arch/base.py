# ------------------------------------------------------------------------------
# Base model for segmentation.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


class BaseSegmentationModel(nn.Module):
    """
    Base class for segmentation models.
    Arguments:
        backbone: A nn.Module of backbone model.
        decoder: A nn.Module of decoder.
    """
    def __init__(self, backbone, decoder):
        super(BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder

    def _init_params(self):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_image_pooling(self, pool_size):
        self.decoder.set_image_pooling(pool_size)

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            result[key] = out
        return result

    def forward(self, x, targets=None):
        input_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)
        pred = self.decoder(features)
        results = self._upsample_predictions(pred, input_shape)

        if targets is None:
            return results
        else:
            return self.loss(results, targets)

    def loss(self, results, targets=None):
        raise NotImplementedError
