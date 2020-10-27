# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
import numpy as np
import torch.nn.functional as F

# get sampling prob


def get_prob(cfg, best_children_pool, CHOICE_NUM=6):
    if cfg.SUPERNET.HOW_TO_PROB == 'even' or (
            cfg.SUPERNET.HOW_TO_PROB == 'teacher' and len(best_children_pool) == 0):
        return None
    elif cfg.SUPERNET.HOW_TO_PROB == 'pre_prob':
        return cfg.SUPERNET.PRE_PROB
    elif cfg.SUPERNET.HOW_TO_PROB == 'teacher':
        op_dict = {}
        for i in range(CHOICE_NUM):
            op_dict[i] = 0
        for item in best_children_pool:
            cand = item[3]
            for block in cand:
                for op in block:
                    op_dict[op] += 1
        sum_op = 0
        for i in range(CHOICE_NUM):
            sum_op = sum_op + op_dict[i]
        prob = []
        for i in range(CHOICE_NUM):
            prob.append(float(op_dict[i]) / sum_op)
        del op_dict, sum_op
        return prob


# sample random architecture
def get_cand_with_prob(CHOICE_NUM, prob=None, sta_num=(4, 4, 4, 4, 4)):
    if prob is None:
        get_random_cand = [
            np.random.choice(
                CHOICE_NUM,
                item).tolist() for item in sta_num]
    else:
        get_random_cand = [
            np.random.choice(
                CHOICE_NUM,
                item,
                prob).tolist() for item in sta_num]
    # print(get_random_cand)
    return get_random_cand


def select_teacher(cfg, best_children_pool, model, random_cand):
    if cfg.SUPERNET.PICK_METHOD == 'top1':
        meta_value, teacher_cand = 0.5, sorted(
            best_children_pool, reverse=True)[0][3]
    elif cfg.SUPERNET.PICK_METHOD == 'meta':
        meta_value, cand_idx, teacher_cand = -1000000000, -1, None
        for now_idx, item in enumerate(best_children_pool):
            inputx = item[4]
            output = F.softmax(model(inputx, random_cand), dim=1)
            weight = model.module.forward_meta(output - item[5])
            if weight > meta_value:
                # deepcopy(torch.nn.functional.sigmoid(weight))
                meta_value = weight
                cand_idx = now_idx
                teacher_cand = best_children_pool[cand_idx][3]
        assert teacher_cand is not None
        meta_value = F.sigmoid(-weight)
    else:
        raise ValueError('Method Not supported')

    return meta_value, teacher_cand
