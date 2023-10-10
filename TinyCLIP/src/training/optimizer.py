from torch import optim
import logging


class EmptyOptimizer:
    def __init__(self):
        self.param_groups = []

    def step(self, *args, **kwargs):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, *args, **kwargs):
        pass

    def zero_grad(self):
        pass


def build_optimizer(args, model):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    # we create three optimizer for image encode, text encoder, and jointly part
    model_parts = [
        list(model.image_named_params()),
        list(model.text_named_params()),
        list(model.joint_named_params()),
    ]

    cnt1 = sum(v.numel() for k, v in named_parameters if v.requires_grad)
    cnt2 = sum(sum(v.numel() for k, v in part if v.requires_grad)
               for part in model_parts)
    assert cnt1 == cnt2, f"cnt1 {cnt1} != cnt2 {cnt2}"

    optimizer = []
    part_names = ['image', 'text', 'joint']
    assert len(model_parts) == len(part_names)
    for name, named_parameters in zip(part_names, model_parts):
        gain_or_bias_params = [p for n, p in named_parameters if exclude(
            n, p) and p.requires_grad and "l0_module" not in n]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad and "l0_module" not in n]
        params_groups = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ]

        num_opt_params = 0
        for pg in params_groups:
            num_opt_params += sum(p.numel() for p in pg['params'])

        logging.info(f'number of optimizer ({name}) params: {num_opt_params}')

        if num_opt_params > 0:
            optimizer_i = optim.AdamW(
                params_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        else:
            optimizer_i = EmptyOptimizer()
        optimizer.append(optimizer_i)

    if args.prune_image or args.prune_text:
        lr_l0 = 0.02
        lr_lamda = args.l0lr
        l0_params = []
        # add l0 optimizer
        if args.prune_image:
            l0_params.extend([
                {
                    "params": [p for n, p in model.image_named_params() if p.requires_grad and "lambda" not in n and "l0_module" in n],
                    "weight_decay": 0.0,
                    "lr": lr_l0
                }, {
                    "params": [p for n, p in model.image_named_params() if p.requires_grad and "lambda" in n and "l0_module" in n],
                    "weight_decay": 0.0,
                    "lr": lr_lamda
                }])
        if args.prune_text:
            l0_params.extend([
                {
                    "params": [p for n, p in model.text_named_params() if p.requires_grad and "lambda" not in n and "l0_module" in n],
                    "weight_decay": 0.0,
                    "lr": lr_l0
                }, {
                    "params": [p for n, p in model.text_named_params() if p.requires_grad and "lambda" in n and "l0_module" in n],
                    "weight_decay": 0.0,
                    "lr": lr_lamda
                }])
        l0_optimizer = optim.AdamW(l0_params)
        optimizer.append(l0_optimizer)

    return optimizer
