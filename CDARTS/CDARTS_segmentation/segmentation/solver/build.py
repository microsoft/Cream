# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/build.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, WarmupPolyLR

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(config):
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = config.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(optimizer_type, gradient_clipper):
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(config, optimizer):
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.
    Args:
        config: configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance
    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not config.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(config.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


def build_optimizer(config, model):
    """Build an optimizer from config.
    Args:
        config: configuration file.
        model: nn.Module, the model.
    Returns:
        A torch Optimizer.
    Raises:
        ValueError: optimizer type has unexpected value.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    # A list of dict: List[Dict[str, Any]].
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = config.SOLVER.BASE_LR
            weight_decay = config.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = config.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = config.SOLVER.BASE_LR * config.SOLVER.BIAS_LR_FACTOR
                weight_decay = config.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if config.SOLVER.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(params, config.SOLVER.BASE_LR, momentum=config.SOLVER.MOMENTUM)
    elif config.SOLVER.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(params, config.SOLVER.BASE_LR, betas=config.SOLVER.ADAM_BETAS,
                                     eps=config.SOLVER.ADAM_EPS)
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.SOLVER.OPTIMIZER))
    optimizer = maybe_add_gradient_clipping(config, optimizer)
    return optimizer


def build_lr_scheduler(config, optimizer):
    """Build a LR scheduler from config.
    Args:
        config: configuration file.
        optimizer: torch optimizer.
    Returns:
        A torch LRScheduler.
    Raises:
        ValueError: LRScheduler type has unexpected value.
    """
    name = config.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            config.SOLVER.STEPS,
            config.SOLVER.GAMMA,
            warmup_factor=config.SOLVER.WARMUP_FACTOR,
            warmup_iters=config.SOLVER.WARMUP_ITERS,
            warmup_method=config.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            config.TRAIN.MAX_ITER,
            warmup_factor=config.SOLVER.WARMUP_FACTOR,
            warmup_iters=config.SOLVER.WARMUP_ITERS,
            warmup_method=config.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            config.TRAIN.MAX_ITER,
            warmup_factor=config.SOLVER.WARMUP_FACTOR,
            warmup_iters=config.SOLVER.WARMUP_ITERS,
            warmup_method=config.SOLVER.WARMUP_METHOD,
            power=config.SOLVER.POLY_LR_POWER,
            constant_ending=config.SOLVER.POLY_LR_CONSTANT_ENDING,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
