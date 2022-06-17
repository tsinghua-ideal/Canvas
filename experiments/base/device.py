import torch


def initialize():
    assert torch.cuda.is_available(), 'No available CUDA devices'
    torch.backends.cudnn.benchmark = True
