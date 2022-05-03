import gc
import torch

from .dataset import get_train_dataset, get_test_dataset, get_single_sample
from .pruner import Pruner


def clean_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def check_cuda_memory(limit: int = 2):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    assert allocated < limit, f'Failed to check CUDA memory, current usage: {allocated}'
