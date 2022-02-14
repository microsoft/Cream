import os
import cv2
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.img_utils import pad_image_to_shape, normalize

logger = get_logger()


class Evaluator(object):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices=0, out_idx=0, threds=5, config=None, logger=None,
                 verbose=False, save_path=None, show_image=False, show_prediction=False):
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices
        if type(self.devices) == int: self.devices = [self.devices]
        self.out_idx = out_idx
        self.threds = threds
        self.config = config
        self.logger = logger

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image
        self.show_prediction = show_prediction

    def run(self, model_path, model_indice, log_file, log_file_link):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            models = [os.path.join(model_path,
                                   'epoch-%s.pth' % model_indice), ]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            result_line, mIoU = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()

    def run_online(self):
        """
        eval during training
        """
        self.val_func = self.network
        result_line, mIoU = self.single_process_evaluation()
        return result_line, mIoU
    
    def single_process_evaluation(self):
        all_results = []
        from pdb import set_trace as bp
        with torch.no_grad():
            for idx in tqdm(range(self.ndata)):
                dd = self.dataset[idx]
                results_dict = self.func_per_iteration(dd, self.devices[0], iter=idx)
                all_results.append(results_dict)
                _, _mIoU = self.compute_metric([results_dict])
        result_line, mIoU = self.compute_metric(all_results)
        return result_line, mIoU

    def run_online_multiprocess(self):
        """
        eval during training
        """
        self.val_func = self.network
        result_line, mIoU = self.multi_process_single_gpu_evaluation()
        return result_line, mIoU

    def multi_process_single_gpu_evaluation(self):
        # start_eval_time = time.perf_counter()
        stride = int(np.ceil(self.ndata / self.threds))

        # start multi-process on single-gpu
        procs = []
        for d in range(self.threds):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[0]
            logger.info('Thread %d handle %d data.' % (d, len(shred_list)))
            p = self.context.Process(target=self.worker, args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line, mIoU = self.compute_metric(all_results)
        # logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mIoU

    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info('GPU %s handle %d data.' % (device, len(shred_list)))
            p = self.context.Process(target=self.worker, args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line, mIoU = self.compute_metric(all_results)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mIoU

    def worker(self, shred_list, device):
        # start_load_time = time.time()
        # logger.info('Load Model on Device %d: %.2fs' % (device, time.time() - start_load_time))
        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device, iter=idx)
            self.results_queue.put(results_dict)

    def func_per_iteration(self, data, device, iter=None):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError

    # evaluate the whole image at once
    def whole_eval(self, img, output_size, input_size=None, device=None):
        if input_size is not None:
            img, margin = self.process_image(img, input_size)
        else:
            img = self.process_image(img, input_size)

        pred = self.val_func_process(img, device)
        if input_size is not None:
            pred = pred[:, margin[0]:(pred.shape[1] - margin[1]),
                   margin[2]:(pred.shape[2] - margin[3])]
        pred = pred.permute(1, 2, 0)
        pred = pred.cpu().numpy()
        if output_size is not None:
            pred = cv2.resize(pred,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        pred = pred.argmax(2)

        return pred

    # slide the window to evaluate the image
    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process(self, img, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data)
                if (isinstance(score, tuple) or isinstance(score, list)) and len(score) > 1:
                    score = score[self.out_idx]
                score = score[0] # a single image pass, ignore batch dim

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)
                # score = score.data

        return score

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img
