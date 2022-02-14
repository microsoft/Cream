# --------------------------------------------------------
# Copyright (c) 2019 Jianyuan Guo (guojianyuan1@huawei.com)
# --------------------------------------------------------

# from .darts_neck_search import DartsNeck
from .hit_neck_search import HitNeck


def build_search_neck(cfg):
    """Build neck model from config dict.
    """
    if cfg is not None:
        cfg_ = cfg.copy()
        neck_type = cfg_.pop('type')
        if neck_type == 'DARTS':
            raise NotImplementedError
            # return DartsNeck(**cfg_)
        elif neck_type == 'HitDet':
            return HitNeck(**cfg_)
        else:
            raise KeyError('Invalid neck type {}'.fromat(neck_type))
    else:
        return None