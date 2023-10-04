import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from open_clip.loss import gather_features, gather_feature
from contextlib import nullcontext
import numpy as np


class ClipSoftLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=None,
            world_size=None,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        if rank is None:
            assert world_size is None
            rank, world_size = dist.get_rank(), dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        assert self.local_loss

        # cache state
        self.feat_buffer = dict()

    def compute_sim(self, image_features, text_features):
        all_image_features = self.gather_feature(image_features)
        all_text_features = self.gather_feature(text_features)

        # calculate logits
        # with torch.cuda.amp.autocast(enabled=False):
        with nullcontext():
            logits_per_image = image_features @ all_text_features.T
            logits_per_text = text_features @ all_image_features.T
        return logits_per_image, logits_per_text

    def gather_feature(self, feat):
        feat_id = id(feat)
        if feat_id not in self.feat_buffer:
            args = (self.local_loss, self.gather_with_grad,
                    self.rank, self.world_size, self.use_horovod)
            all_feat = gather_feature(feat, *args)
            self.feat_buffer[feat_id] = all_feat
        return self.feat_buffer[feat_id]

    def forward(self,
                image_features, text_features, logit_scale,
                teacher_image_features, teacher_text_features, teacher_logit_scale,
                average_two_losses=True,
                labels=None,
                ):
        # calculated ground-truth and cache if enabled
        logits_per_image, logits_per_text = self.compute_sim(
            image_features, text_features)
        teacher_logits_per_image, teacher_logits_per_text = self.compute_sim(
            teacher_image_features, teacher_text_features)

        self.feat_buffer.clear()

        # with torch.cuda.amp.autocast(enabled=False):
        with nullcontext():
            logits_per_image = logit_scale * logits_per_image
            logits_per_text = logit_scale * logits_per_text
            teacher_logits_per_image = teacher_logit_scale * teacher_logits_per_image
            teacher_logits_per_text = teacher_logit_scale * teacher_logits_per_text

            def single_loss_fn(logits, teacher_logits):
                teacher_probs = F.softmax(teacher_logits, -1)
                return F.cross_entropy(logits, teacher_probs)

            if average_two_losses:
                total_loss = (single_loss_fn(logits_per_image, teacher_logits_per_image) +
                              single_loss_fn(logits_per_text, teacher_logits_per_text)) / 2
                return total_loss
            else:
                img2text_loss = single_loss_fn(
                    logits_per_image, teacher_logits_per_image)
                text2img_loss = single_loss_fn(
                    logits_per_text, teacher_logits_per_text)
                return img2text_loss, text2img_loss
