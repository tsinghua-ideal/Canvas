import torch


def check_available():
    assert torch.cuda.is_available(), 'No available CUDA devices'
