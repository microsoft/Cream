import json
import logging
import os

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from collections import UserDict

try:
    import wandb
except ImportError:
    wandb = None

from timm.utils.model import unwrap_model
from .distributed import is_master
from .zero_shot import zero_shot_eval


def evaluate(model, data, epoch, args, tb_writer=None, step=None, num_feed_images=None):
    metrics = {}
    models = [model]
    names = ['']
    assert len(names) == len(models)
    for name, model_i in zip(names, models):
        model_i.eval()
        zero_shot_metrics = zero_shot_eval(model_i, data, epoch, args)
        zero_shot_metrics = dict((name + k, v)
                                 for k, v in zero_shot_metrics.items())
        metrics.update(zero_shot_metrics)

    if not metrics:
        return metrics

    if not is_master(args):
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            log = {f"val/{name}": val, 'epoch': epoch}
            extra_kwargs = dict()
            if step is not None:
                log['step'] = step
                extra_kwargs['step'] = step
            if num_feed_images is not None:
                log['num_feed_images'] = num_feed_images
            wandb.log(log, **extra_kwargs)
    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
