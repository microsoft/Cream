# ------------------------------------------------------------------------------
# Generates the correct format for official evaluation code.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import numpy as np


def get_cityscapes_instance_format(panoptic, sem, ctr_hmp, label_divisor, score_type="semantic"):
    """
    Get Cityscapes instance segmentation format.
    Arguments:
        panoptic: A Numpy Ndarray of shape [H, W].
        sem: A Numpy Ndarray of shape [C, H, W] of raw semantic output.
        ctr_hmp: A Numpy Ndarray of shape [H, W] of raw center heatmap output.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        score_type: A string, how to calculates confidence scores for instance segmentation.
            - "semantic": average of semantic segmentation confidence within the instance mask.
            - "instance": confidence of heatmap at center point of the instance mask.
            - "both": multiply "semantic" and "instance".
    Returns:
        A List contains instance segmentation in Cityscapes format.
    """
    instances = []

    pan_labels = np.unique(panoptic)
    for pan_lab in pan_labels:
        if pan_lab % label_divisor == 0:
            # This is either stuff or ignored region.
            continue

        ins = OrderedDict()

        train_class_id = pan_lab // label_divisor
        ins['pred_class'] = train_class_id

        mask = panoptic == pan_lab
        ins['pred_mask'] = np.array(mask, dtype='uint8')

        sem_scores = sem[train_class_id, ...]
        ins_score = np.mean(sem_scores[mask])
        # mask center point
        mask_index = np.where(panoptic == pan_lab)
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        ctr_score = ctr_hmp[int(center_y), int(center_x)]

        if score_type == "semantic":
            ins['score'] = ins_score
        elif score_type == "instance":
            ins['score'] = ctr_score
        elif score_type == "both":
            ins['score'] = ins_score * ctr_score
        else:
            raise ValueError("Unknown confidence score type: {}".format(score_type))

        instances.append(ins)

    return instances
