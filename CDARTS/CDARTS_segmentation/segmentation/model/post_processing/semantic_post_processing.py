# ------------------------------------------------------------------------------
# Post-processing to get semantic segmentation results.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch

__all__ = ['get_semantic_segmentation']


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
