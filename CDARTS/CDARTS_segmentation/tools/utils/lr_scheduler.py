import math


class Iter_LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, args, max_iteration, iters_per_epoch):
        self.mode = args.mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = args.base_lr
        self.lr_step = args.lr_step
        self.iters_per_epoch = iters_per_epoch
        self.max_iteration = max_iteration
        self.epoch = -1
        self.warmup_iters = args.warmup_iters
        self.min_lr = args.min_lr if args.min_lr is not None else 0
        self.warmup_start_lr = args.warmup_start_lr
        self.warmup_factor = (self.lr / args.warmup_start_lr) ** (1. / args.warmup_iters)

    def __call__(self, optimizer, iteration):
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            lr = self.warmup_start_lr * (self.warmup_factor ** iteration)
        elif self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * iteration / self.max_iteration * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - (iteration - self.warmup_iters) / (self.max_iteration - self.warmup_iters)), 0.9)
        elif self.mode == 'step':  # TODO: Fix the step mode
            print('Warning! Now the step decline lr exists some issue')
            if not self.lr_step:
                raise NotImplementedError
            epoch = iteration // self.iters_per_epoch
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if iteration == self.warmup_iters:
            print('==> warmup done, start to implement poly lr strategy')
        if (not iteration % self.iters_per_epoch) and (iteration // self.iters_per_epoch > self.epoch):
            epoch = iteration // self.iters_per_epoch
            print('\n=>Epoches %i, learning rate = %.4f' % (epoch, lr))
            self.epoch = epoch

        optimizer.param_groups[0]['lr'] = max(lr, self.min_lr)

    def get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']