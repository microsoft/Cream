# ------------------------------------------------------------------------------
# Adds `segmentation` package into Python path.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..')
add_path(lib_path)
add_path(this_dir)
add_path(osp.join(lib_path, 'tools'))
