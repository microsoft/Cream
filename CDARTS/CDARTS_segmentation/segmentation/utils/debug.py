# ------------------------------------------------------------------------------
# Saves raw outputs and targets.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os

import numpy as np
import PIL.Image as img

import torch

from .save_annotation import label_to_color_image
from .flow_vis import flow_compute_color


def save_debug_images(dataset, batch_images, batch_targets, batch_outputs, out_dir=None, iteration=0,
                      target_keys=('semantic', 'center', 'offset', 'center_weights', 'offset_weights'),
                      output_keys=('semantic', 'center', 'offset'),
                      iteration_to_remove=-1, is_train=True):
    """Saves a mini-batch of images for debugging purpose.
        - image: the augmented input image
        - label: the augmented labels including
            - semantic: semantic segmentation label
            - center: center heatmap
            - offset: offset field
            - instance_ignore_mask: ignore mask
        - prediction: the raw output of the model (without post-processing)
            - semantic: semantic segmentation label
            - center: center heatmap
            - offset: offset field
    Args:
        dataset: The Dataset.
        batch_images: Tensor of shape [N, 3, H, W], a batch of input images.
        batch_targets: Dict, a dict containing batch of targets.
            - semantic: a Tensor of shape [N, H, W]
            - center: a Tensor of shape [N, 1, H, W]
            - offset: a Tensor of shape [N, 2, H, W]
            - semantic_weights: a Tensor of shape [N, H, W]
            - center_weights: a Tensor of shape [N, H, W]
            - offset_weights: a Tensor of shape [N, H, W]
        batch_outputs: Dict, a dict containing batch of outputs.
            - semantic: a Tensor of shape [N, H, W]
            - center: a Tensor of shape [N, 1, H, W]
            - offset: a Tensor of shape [N, 2, H, W]
        out_dir: String, the directory to which the results will be saved.
        iteration: Integer, iteration number.
        target_keys: List, target keys to save.
        output_keys: List, output keys to save.
        iteration_to_remove: Integer, iteration number to remove.
        is_train: Boolean, save train or test debugging image.
    """
    batch_size = batch_images.size(0)
    map_height = batch_images.size(2)
    map_width = batch_images.size(3)

    grid_image = np.zeros(
        (map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    num_targets = len(target_keys)
    grid_target = np.zeros(
        (num_targets * map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    num_outputs = len(output_keys)
    grid_output = np.zeros(
        (num_outputs * map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    semantic_pred = torch.argmax(batch_outputs['semantic'].detach(), dim=1)
    if 'foreground' in batch_outputs:
        foreground_pred = torch.argmax(batch_outputs['foreground'].detach(), dim=1)
    else:
        foreground_pred = None

    for i in range(batch_size):
        width_begin = map_width * i
        width_end = map_width * (i + 1)

        # save images
        image = dataset.reverse_transform(batch_images[i])
        grid_image[:, width_begin:width_end, :] = image

        if 'semantic' in target_keys:
            # save gt semantic
            gt_sem = batch_targets['semantic'][i].cpu().numpy()
            gt_sem = label_to_color_image(gt_sem, dataset.create_label_colormap())
            grid_target[:map_height, width_begin:width_end, :] = gt_sem

        if 'center' in target_keys:
            # save gt center
            gt_ctr = batch_targets['center'][i].squeeze().cpu().numpy()
            gt_ctr = gt_ctr[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
            gt_ctr = gt_ctr.clip(0, 255)
            # gt_ctr = 0.7 * gt_ctr + (1 - 0.3) * image
            grid_target[map_height:2 * map_height, width_begin:width_end, :] = gt_ctr

        if 'offset' in target_keys:
            # save gt offset
            gt_off = batch_targets['offset'][i].permute(1, 2, 0).cpu().numpy()
            gt_off = flow_compute_color(gt_off[:, :, 1], gt_off[:, :, 0])
            grid_target[2 * map_height:3 * map_height, width_begin:width_end, :] = gt_off

        if 'semantic_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['semantic_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] / np.max(gt_ign) * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[3 * map_height:4 * map_height, width_begin:width_end, :] = gt_ign

        if 'center_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['center_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[4 * map_height:5 * map_height, width_begin:width_end, :] = gt_ign

        if 'offset_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['offset_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[5 * map_height:6 * map_height, width_begin:width_end, :] = gt_ign

        if 'foreground' in target_keys:
            # save gt foreground
            gt_fg = batch_targets['foreground'][i].cpu().numpy()
            gt_fg = gt_fg[:, :, None] * 255
            grid_target[6 * map_height:7 * map_height, width_begin:width_end, :] = gt_fg

        if 'semantic' in output_keys:
            # save pred semantic
            pred_sem = semantic_pred[i].cpu().numpy()
            pred_sem = label_to_color_image(pred_sem, dataset.create_label_colormap())
            grid_output[:map_height, width_begin:width_end, :] = pred_sem

        if 'center' in output_keys:
            # save pred center
            pred_ctr = batch_outputs['center'][i].detach().squeeze().cpu().numpy()
            pred_ctr = pred_ctr[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
            pred_ctr = pred_ctr.clip(0, 255)
            # pred_ctr = 0.7 * pred_ctr + (1 - 0.3) * image
            grid_output[map_height:2 * map_height, width_begin:width_end, :] = pred_ctr

        if 'offset' in output_keys:
            # save pred offset
            pred_ctr = batch_outputs['offset'][i].detach().permute(1, 2, 0).cpu().numpy()
            pred_ctr = flow_compute_color(pred_ctr[:, :, 1], pred_ctr[:, :, 0])
            grid_output[2 * map_height:3 * map_height, width_begin:width_end, :] = pred_ctr

        if 'foreground' in output_keys:
            # save pred foreground
            if foreground_pred is not None:
                pred_fg = foreground_pred[i].cpu().numpy()
                pred_fg = pred_fg[:, :, None] * 255
                grid_output[3 * map_height:4 * map_height, width_begin:width_end, :] = pred_fg

    if out_dir is not None:
        if is_train:
            pil_image = img.fromarray(grid_image.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_target.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_output.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
        else:
            pil_image = img.fromarray(grid_image.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_test_images', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            if grid_target.size:
                pil_image = img.fromarray(grid_target.astype(dtype=np.uint8))
                with open('%s/%s_%d.png' % (out_dir, 'debug_test_targets', iteration), mode='wb') as f:
                    pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_output.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_test_outputs', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')

    if is_train:
        if iteration_to_remove >= 0:
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration_to_remove))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration_to_remove))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration_to_remove))
            # 0 is a special iter
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_images', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_images', 0))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', 0))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', 0))
