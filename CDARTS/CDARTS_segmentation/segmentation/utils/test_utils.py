# ------------------------------------------------------------------------------
# Utility functions for multi-scale testing.
# Written by Pingjun (https://github.com/bowenc0221/panoptic-deeplab/issues/25)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
import cv2
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import segmentation.data.transforms.transforms as T


def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def upsample_predictions(pred, input_shape,scale):
    # Override upsample method to correctly handle `offset`
    result = OrderedDict()
    for key in pred.keys():
        out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
        if 'offset' in key:  #The order of second dim is (offset_y, offset_x)
            out *= 1.0 / scale
        result[key] = out
    return result

def get_semantic_segmentation(sem):
    """
    Post-processing for semantic segmentation branch.
    Arguments:
        sem: A Tensor of shape [N, C, H, W], where N is the batch size, for consistent, we only
            support N=1.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)
    return torch.argmax(sem, dim=0, keepdim=True)

def multi_scale_inference(config, model, raw_image, t_image, device):
    scales = config.TEST.SCALE_LIST
    flip = config.TEST.FLIP_TEST
    # output_stride = 2 ** (5 - sum(config.MODEL.BACKBONE.DILATION))
    # train_crop_h, train_crop_w = config.TEST.CROP_SIZE
    # scale = 1. / output_stride
    # pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
    # pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)
    # transforms
    transforms = T.Compose(
        [  
            T.ToTensor(),
            T.Normalize(config.DATASET.MEAN, config.DATASET.STD)
        ]
    )
    if flip:
        flip_range = 2
    else:
        flip_range = 1
    
    # h,w,_ = raw_image.shape
    _, _, h, w = t_image.shape
    org_h_pad = (h + 31) // 32 * 32
    org_w_pad = (w + 31) // 32 * 32

    sum_semantic_with_flip = 0
    sum_center_with_flip = 0
    sum_offset_with_flip = 0

    for i in range(len(scales)):
        image = raw_image
        scale = scales[i]
        raw_h = int(h * scale)
        raw_w = int(w * scale)

        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).astype(np.int32)
        nh,nw,_ = image.shape

        # pad image
        new_h = (raw_h + 31) // 32 * 32
        new_w = (raw_w + 31) // 32 * 32
        input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        input_image[:, :] = config.DATASET.MEAN
        # input_image[:raw_h, :raw_w, :] = image
        input_image[:nh, :nw, :] = image

        image, _ = transforms(input_image, None)
        image = image.unsqueeze(0).to(device)


        model = model.to(device)
        
        for flip in range(flip_range):
            if flip:
                image = flip_tensor(image, 3)
            out_dict = model(image)
            for key in out_dict.keys():  # return to raw_input shape
                out_dict[key] = out_dict[key][:, :, : raw_h, : raw_w]

            if raw_h != org_h_pad or raw_w != org_w_pad:
                out_dict = upsample_predictions(out_dict, (org_h_pad, org_w_pad), scale)

            # average softmax or logit?
            semantic_pred = out_dict['semantic']
            # semantic_pred = F.softmax(out_dict['semantic'],dim=1)

            center_pred = out_dict['center']
            offset_pred = out_dict['offset']
            if flip:
                semantic_pred = flip_tensor(semantic_pred,3)
                center_pred = flip_tensor(center_pred,3)
                offset_pred = flip_tensor(offset_pred,3)
                offset_pred[:, 1, :, :] *= (-1)

            sum_semantic_with_flip += semantic_pred
            sum_center_with_flip += center_pred
            sum_offset_with_flip += offset_pred

    semantic_mean = sum_semantic_with_flip / (flip_range * len(scales))
    center_mean = sum_center_with_flip / (flip_range * len(scales))
    offset_mean = sum_offset_with_flip / (flip_range * len(scales))

    out_dict['semantic'] = semantic_mean
    out_dict['center'] = center_mean
    out_dict['offset'] = offset_mean
    return out_dict
