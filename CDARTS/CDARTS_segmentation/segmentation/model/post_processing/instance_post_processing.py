# ------------------------------------------------------------------------------
# Post-processing to get instance and panoptic segmentation results.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from .semantic_post_processing import get_semantic_segmentation

__all__ = ['find_instance_center', 'get_instance_segmentation', 'get_panoptic_segmentation']


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_instance_segmentation(sem_seg, ctr_hmp, offsets, thing_list, threshold=0.1, nms_kernel=3, top_k=None,
                              thing_seg=None):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if thing_seg is None:
        # gets foreground segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in thing_list:
            thing_seg[sem_seg == thing_class] = 1

    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    ins_seg = group_pixels(ctr, offsets)
    return thing_seg * ins_seg, ctr.unsqueeze(0)


def merge_semantic_and_instance(sem_seg, ins_seg, label_divisor, thing_list, stuff_area, void_label):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # In case thing mask does not align with semantic prediction
    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1

    # keep track of instance id for each class
    class_id_tracker = {}

    # paste thing by majority voting
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1
        class_id_tracker[class_id.item()] += 1
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_panoptic_segmentation(sem, ctr_hmp, offsets, thing_list, label_divisor, stuff_area, void_label,
                              threshold=0.1, nms_kernel=3, top_k=None, foreground_mask=None):
    """
    Post-processing for panoptic segmentation.
    Arguments:
        sem: A Tensor of shape [N, C, H, W] of raw semantic output, where N is the batch size, for consistent,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        foreground_mask: A Tensor of shape [N, 2, H, W] of raw foreground mask, where N is the batch size,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel), int64.
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.dim() != 4 and sem.dim() != 3:
        raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    if sem.dim() == 4 and sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if foreground_mask is not None:
        if foreground_mask.dim() != 4 and foreground_mask.dim() != 3:
            raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    if sem.dim() == 4:
        semantic = get_semantic_segmentation(sem)
    else:
        semantic = sem

    if foreground_mask is not None:
        if foreground_mask.dim() == 4:
            thing_seg = get_semantic_segmentation(foreground_mask)
        else:
            thing_seg = foreground_mask
    else:
        thing_seg = None

    instance, center = get_instance_segmentation(semantic, ctr_hmp, offsets, thing_list,
                                                 threshold=threshold, nms_kernel=nms_kernel, top_k=top_k,
                                                 thing_seg=thing_seg)
    panoptic = merge_semantic_and_instance(semantic, instance, label_divisor, thing_list, stuff_area, void_label)

    return panoptic, center
