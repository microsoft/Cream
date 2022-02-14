# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/sem_seg_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
from collections import OrderedDict

import numpy as np

from fvcore.common.file_io import PathManager
from segmentation.utils import save_annotation


class SemanticEvaluator:
    """
    Evaluate semantic segmentation
    """
    def __init__(self, num_classes, ignore_label=255, output_dir=None, train_id_to_eval_id=None):
        """
        Args:
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
        """
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1  # store ignore label in the last class
        self._train_id_to_eval_id = train_id_to_eval_id

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
            train_id_to_eval_id (list): maps training id to evaluation id.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction

    def update(self, pred, gt, image_filename=None):
        pred = pred.astype(np.int)
        gt = gt.astype(np.int)
        gt[gt == self._ignore_label] = self._num_classes

        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)

        if self._output_dir:
            if self._train_id_to_eval_id is not None:
                pred = self._convert_train_id_to_eval_id(pred, self._train_id_to_eval_id)
            if image_filename is None:
                raise ValueError('Need to provide filename to save.')
            save_annotation(
                pred, self._output_dir, image_filename, add_colormap=False)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
