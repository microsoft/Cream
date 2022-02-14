#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
cv2.setNumThreads(0)
import numpy as np

from utils.visualize import print_iou, show_img, show_prediction
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score

logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, iter=None):
        if self.config is not None: config = self.config
        img = data['data']
        label = data['label']
        name = data['fn']

        if len(config.eval_scale_array) == 1:
            pred = self.whole_eval(img, None, device)
        else:
            pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)
        
        # tensorboard logger does not fit multiprocess
        if self.logger is not None and iter is not None:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean, label, pred)
            self.logger.add_image('vis', np.swapaxes(np.swapaxes(comp_img, 0, 2), 1, 2), iter)

        if self.show_image or self.show_prediction:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            if self.show_image:
                comp_img = show_img(colors, config.background, image, clean, label, pred)
            else:
                comp_img = show_prediction(colors, config.background, image, pred)
            cv2.imwrite(name + ".png", comp_img[:,:,::-1])

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, mean_IU_no_back, mean_pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iu, mean_pixel_acc, self.dataset.get_class_names(), True)
        return result_line, mean_IU
