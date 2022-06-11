from torch import optim

from timm.scheduler import create_scheduler


def get_scheduler(args, optimizer: optim.Optimizer):
    return create_scheduler(args, optimizer)
