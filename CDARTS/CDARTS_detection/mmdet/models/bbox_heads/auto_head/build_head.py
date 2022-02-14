# --------------------------------------------------------
# Copyright (c) 2019 Jianyuan Guo (guojianyuan1@huawei.com)
# --------------------------------------------------------

# from .darts_head_search import DartsHead
from .mbblock_head_search import MbblockHead


def build_search_head(cfg):
    """Build head model from config dict.
    """
    if cfg is not None:
        cfg_ = cfg.copy()
        head_type = cfg_.pop('type')
        if head_type == 'DARTS':
            raise NotImplementedError
        elif head_type == 'MBBlock':
            return MbblockHead(**cfg_)
        else:
            raise KeyError('Invalid head type {}'.fromat(head_type))
    else:
        return None