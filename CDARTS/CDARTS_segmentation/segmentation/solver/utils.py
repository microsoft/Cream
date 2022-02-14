# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/hooks.py#L195
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import Counter


def get_lr_group_id(optimizer):
    """
    Returns the group id with majority of lr.
    """
    # Get the correct parameter group id to access to lr info.
    largest_group = max(len(g["params"]) for g in optimizer.param_groups)
    if largest_group == 1:
        # If all groups have one parameter,
        # then find the most common initial LR, and use it for summary
        lr_count = Counter([g["lr"] for g in optimizer.param_groups])
        lr = lr_count.most_common()[0][0]
        for i, g in enumerate(optimizer.param_groups):
            if g["lr"] == lr:
                best_param_group_id = i
                break
    else:
        for i, g in enumerate(optimizer.param_groups):
            if len(g["params"]) == largest_group:
                best_param_group_id = i
                break
    return best_param_group_id
