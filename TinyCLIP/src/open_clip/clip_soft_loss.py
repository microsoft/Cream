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
            mask_diag=None,
            use_top2=False,
            ignore_diag=False,
            select_topk=None,
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
        self.mask_diag = mask_diag
        self.use_top2 = use_top2
        self.select_topk = select_topk
        self.ignore_diag = ignore_diag
        assert self.local_loss

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.feat_buffer = dict()

        self.stat = None

    def compute_sim(self, image_features, text_features):
        all_image_features = self.gather_feature(image_features)
        all_text_features = self.gather_feature(text_features)

        # calculate logits
        # with torch.cuda.amp.autocast(enabled=False):
        with nullcontext():
            logits_per_image = image_features @ all_text_features.T
            logits_per_text = text_features @ all_image_features.T
        return logits_per_image, logits_per_text

    @torch.no_grad()
    def compute_label_logits(self, labels, ignore_image_text=False):
        # minus value for image-text data
        # label = -1, image-text sample
        # label >= 0: imagenet labels
        # labels: [-1, -1, 0, 3], [-1, -1, 1, 3]
        batch_size = len(labels)
        start = -(batch_size * self.rank + 1)
        # image-text mask
        # [True, True, False, False]
        minus_mask = labels < 0
        # number of image-text samples
        minus_num = torch.count_nonzero(minus_mask).item()
        dtype = labels.dtype
        device = labels.device
        # [-1, -2, 0, 3], [-5, -6, 1, 3]
        labels[minus_mask] = torch.arange(start, start - minus_num, -1, dtype=dtype, device=device)
        args = (self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
        # [-1, -2, 0, 3, -5, -6, 1, 3]
        all_labels = gather_feature(labels, *args)
        '''
            [-1, -2, 0, 3, -5, -6, 1, 3]
        -1    T
        -2        T
         0           T
         3              T             T
         --------------------------------
        -5                  T
        -6                      T
         1                         T
         3              T             T
        '''
        mask = (labels.view(-1, 1) == all_labels.view(1, -1))
        if ignore_image_text:
            mask[minus_mask] = 0
        return mask.float()

    def gather_feature(self, feat):
        feat_id = id(feat)
        if feat_id not in self.feat_buffer:
            args = (self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            all_feat = gather_feature(feat, *args)
            self.feat_buffer[feat_id] = all_feat
        return self.feat_buffer[feat_id]

    def mask_diag_(self, feat, value):
        H, W = feat.shape
        assert H * self.world_size == W
        start = H * self.rank
        feat[:, start:start+H].fill_diagonal_(value)

    def compute_top2_probs(self, feat, value):
        self.mask_diag_(feat, value)
        feat = F.softmax(feat, -1)
        self.mask_diag_(feat, 1)
        return feat

    def forward(self,
                image_features, text_features, logit_scale,
                teacher_image_features, teacher_text_features, teacher_logit_scale,
                labels=None,
                unicl=0.0,
                average_two_losses=True,
                ):
        # calculated ground-truth and cache if enabled
        logits_per_image, logits_per_text = self.compute_sim(image_features, text_features)
        teacher_logits_per_image, teacher_logits_per_text = self.compute_sim(teacher_image_features, teacher_text_features)

        self.feat_buffer.clear()

        # with torch.cuda.amp.autocast(enabled=False):
        with nullcontext():
            logits_per_image = logit_scale * logits_per_image
            logits_per_text = logit_scale * logits_per_text
            teacher_logits_per_image = teacher_logit_scale * teacher_logits_per_image
            teacher_logits_per_text = teacher_logit_scale * teacher_logits_per_text

            if unicl > 0.0:
                # 0~1
                label_logits = self.compute_label_logits(labels, ignore_image_text=(unicl >= 100.0))
            else:
                label_logits = None

            if self.mask_diag is not None:
                self.mask_diag_(logits_per_image, self.mask_diag)
                self.mask_diag_(logits_per_text, self.mask_diag)
                self.mask_diag_(teacher_logits_per_image, self.mask_diag)
                self.mask_diag_(teacher_logits_per_text, self.mask_diag)

            def single_loss_fn(logits, teacher_logits, name):
                if self.ignore_diag:
                    self.mask_diag_(logits, -100)
                    self.mask_diag_(teacher_logits, -100)
                if self.use_top2:
                    self.mask_diag_(teacher_logits, -100)

                if self.select_topk is not None:
                    topk = min(self.select_topk, teacher_logits.size(1))
                    ind = teacher_logits.topk(dim=1, k=topk, largest=True, sorted=False).indices
                    teacher_logits = teacher_logits.gather(1, ind)
                    logits = logits.gather(1, ind)

                teacher_probs = F.softmax(teacher_logits, -1)

                if self.use_top2:
                    self.mask_diag_(teacher_probs, 1)
                    raise NotImplementedError('[TODO] the sample loss should be divided by the sum(teacher_probs)')

                if unicl > 0.0 and unicl <= 1.0:
                    return F.cross_entropy(logits, teacher_probs) * (1.0 - unicl) + \
                            (F.cross_entropy(logits, label_logits, reduction='none') / label_logits.sum(-1)).mean() * unicl
                elif unicl >= 100.0:
                    teacher_probs = torch.maximum(teacher_probs, label_logits)
                    return (F.cross_entropy(logits, teacher_probs, reduction='none') / teacher_probs.sum(-1)).mean()

                if self.stat is not None and len(self.stat) == 0:
                    bins = 50
                    hist_range = (0, 1)
                    student_probs = F.softmax(logits, -1)
                    student_probs_hist = torch.histogram(student_probs.detach().cpu(), bins=bins, range=hist_range)[0]
                    teacher_probs_hist = torch.histogram(teacher_probs.detach().cpu(), bins=bins, range=hist_range)[0]
                    student_probs_hist = student_probs_hist.cuda()
                    teacher_probs_hist = teacher_probs_hist.cuda()
                    # dist.all_reduce(student_probs_hist, dist.ReduceOp.SUM)
                    # dist.all_reduce(teacher_probs_hist, dist.ReduceOp.SUM)

                    student_probs_hist = student_probs_hist.cpu().numpy()
                    teacher_probs_hist = teacher_probs_hist.cpu().numpy()

                    self.stat[f'{name}_student_prob_hist'] = student_probs_hist
                    self.stat[f'{name}_teacher_prob_hist'] = teacher_probs_hist

                    xs = np.linspace(0, 1, bins+1)[:-1]

                    def _stat_hist(hist, sub_name):
                        assert len(hist) == len(xs), (len(hist), len(xs))
                        n_elems = hist.sum()
                        _mean = (hist * xs).sum() / n_elems
                        _s2mean = (hist * np.square(xs)).sum() / n_elems
                        self.stat[f'{name}_{sub_name}_hist_mean'] = _mean
                        self.stat[f'{name}_{sub_name}_hist_std'] = np.sqrt(_s2mean - np.square(_mean))

                    _stat_hist(student_probs_hist, 'student')
                    _stat_hist(teacher_probs_hist, 'teacher')

                    def _stat_diag(probs, sub_name):
                        diag = probs[:, self.rank*len(probs):].diag()
                        fn_names = ['min', 'max', 'mean']
                        ops = [dist.ReduceOp.MIN, dist.ReduceOp.MAX, dist.ReduceOp.SUM]
                        for fn_name, op in zip(fn_names, ops):
                            v = getattr(diag, fn_name)()
                            # dist.all_reduce(v, op=op)
                            self.stat[f'{name}_{sub_name}_{fn_name}'] = v.item()
                        # self.stat[f'{name}_{sub_name}_mean'] /= self.world_size
                        s = torch.stack([diag.square().mean(), diag.mean()])
                        # dist.all_reduce(s, op=dist.ReduceOp.SUM)
                        # s /= self.world_size
                        self.stat[f'{name}_{sub_name}_std'] = (s[0] - s[1].square()).sqrt().item()

                    _stat_diag(student_probs, 'diag_student_prob')
                    _stat_diag(teacher_probs, 'diag_teacher_prob')
                    _stat_diag(logits, 'diag_student_logit')
                    _stat_diag(teacher_logits, 'diag_teacher_logit')


                return F.cross_entropy(logits, teacher_probs)

            if average_two_losses:
                total_loss = (single_loss_fn(logits_per_image, teacher_logits_per_image, 'image') +
                              single_loss_fn(logits_per_text, teacher_logits_per_text, 'text')) / 2
                return total_loss
            else:
                img2text_loss = single_loss_fn(logits_per_image, teacher_logits_per_image, 'image')
                text2img_loss = single_loss_fn(logits_per_text, teacher_logits_per_text, 'text')
                return img2text_loss, text2img_loss
