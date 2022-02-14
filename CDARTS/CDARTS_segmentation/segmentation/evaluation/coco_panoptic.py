# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/panoptic_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
from collections import OrderedDict
import os
import json

import numpy as np
from tabulate import tabulate

from fvcore.common.file_io import PathManager
from segmentation.utils import save_annotation

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator:
    """
    Evaluate panoptic segmentation
    """
    def __init__(self, output_dir=None, train_id_to_eval_id=None, label_divisor=256, void_label=65280,
                 gt_dir='./datasets/coco', split='val2017', num_classes=133):
        """
        Args:
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            label_divisor (int):
            void_label (int):
            gt_dir (str): path to ground truth annotations.
            split (str): evaluation split.
            num_classes (int): number of classes.
        """
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._panoptic_dir = os.path.join(self._output_dir, 'predictions')
        if self._panoptic_dir:
            PathManager.mkdirs(self._panoptic_dir)

        self._predictions = []
        self._predictions_json = os.path.join(output_dir, 'predictions.json')

        self._train_id_to_eval_id = train_id_to_eval_id
        self._label_divisor = label_divisor
        self._void_label = void_label
        self._num_classes = num_classes

        self._logger = logging.getLogger(__name__)

        self._gt_json_file = os.path.join(gt_dir, 'annotations', 'panoptic_{}.json'.format(split))
        self._gt_folder = os.path.join(gt_dir, 'annotations', 'panoptic_{}'.format(split))
        self._pred_json_file = os.path.join(output_dir, 'predictions.json')
        self._pred_folder = self._panoptic_dir

    def update(self, panoptic, image_filename=None, image_id=None):
        from panopticapi.utils import id2rgb

        if image_filename is None:
            raise ValueError('Need to provide image_filename.')
        if image_id is None:
            raise ValueError('Need to provide image_id.')

        # Change void region.
        panoptic[panoptic == self._void_label] = 0

        segments_info = []
        for pan_lab in np.unique(panoptic):
            pred_class = pan_lab // self._label_divisor
            if self._train_id_to_eval_id is not None:
                pred_class = self._train_id_to_eval_id[pred_class]

            segments_info.append(
                {
                    'id': int(pan_lab),
                    'category_id': int(pred_class),
                }
            )

        save_annotation(id2rgb(panoptic), self._panoptic_dir, image_filename, add_colormap=False)
        self._predictions.append(
            {
                'image_id': int(image_id),
                'file_name': image_filename + '.png',
                'segments_info': segments_info,
            }
        )

    def evaluate(self):
        from panopticapi.evaluation import pq_compute

        gt_json_file = self._gt_json_file
        gt_folder = self._gt_folder
        pred_json_file = self._pred_json_file
        pred_folder = self._pred_folder

        with open(gt_json_file, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with PathManager.open(self._predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        pq_res = pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder)

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        self._logger.info(results)
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)
