from torch import nn

from timm.optim import create_optimizer_v2


def get_optimizer(args, model: nn.Module):
    return create_optimizer_v2(model,
                               opt=args.opt, lr=args.lr,
                               weight_decay=args.weight_decay,
                               momentum=args.momentum,
                               eps=args.opt_eps, betas=args.opt_betas)
