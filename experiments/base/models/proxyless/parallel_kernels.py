import numpy as np
import canvas
import torch
import torch.nn.functional as F
from functools import partial
from torch import nn


class ParallelKernels(nn.Module):
    def __init__(self, kernel_cls_list, **kwargs):
        super().__init__()
        assert len(kernel_cls_list) > 1
        self.module_list = nn.ModuleList([kernel_cls(*kwargs.values()) for kernel_cls in kernel_cls_list])
        self.alphas = nn.Parameter(torch.ones(len(kernel_cls_list)))
        self.bin_alphas = nn.Parameter(torch.Tensor(len(self)))
        self.active_only = None
        self.module_list_backup = None
        self.active_idx, self.inactive_idx = None, None
        self.old_active_alpha, self.old_inactive_alpha = None, None

    def get_softmaxed_alphas(self):
        return F.softmax(self.alphas, dim=0)

    def set_indices(self, active_idx, inactive_idx):
        self.active_only = inactive_idx is None
        self.active_idx, self.inactive_idx = active_idx, inactive_idx
        self.bin_alphas.data.zero_()
        self.bin_alphas.data[active_idx] = 1
        assert self.module_list_backup is None
        self.module_list_backup = []
        for i in range(len(self)):
            if i not in (active_idx, inactive_idx):
                self.module_list_backup.append(self.module_list[i])
                self.module_list[i] = None
            else:
                self.module_list_backup.append(None)

    def restore_all(self):
        assert self.module_list_backup is not None
        for i in range(len(self)):
            if i not in (self.active_idx, self.inactive_idx):
                assert self.module_list[i] is None
                self.module_list[i] = self.module_list_backup[i]
            assert self.module_list[i] is not None
        self.active_idx, self.inactive_idx = None, None
        self.module_list_backup = None

    def __len__(self):
        return len(self.module_list)

    def forward(self, x):
        assert self.active_idx is not None
        active_module = self.module_list[self.active_idx]
        if self.active_only:
            return active_module(x)
        else:
            assert self.inactive_idx is not None
            inactive_module = self.module_list[self.inactive_idx]
            active = self.bin_alphas[self.active_idx] * active_module(x)
            inactive = self.bin_alphas[self.inactive_idx] * inactive_module(x).detach()
            return active + inactive

    def set_alpha_grad(self):
        # TODO: implement GPU version
        bin_alpha_grads = self.bin_alphas.grad
        assert self.alphas.grad is None
        self.alphas.grad = torch.zeros_like(self.alphas)

        assert not self.active_only
        probs = F.softmax(torch.Tensor([self.alphas[self.active_idx], self.alphas[self.inactive_idx]]), dim=0)
        for i in range(2):
            for j in range(2):
                original_i = self.active_idx if i == 0 else self.inactive_idx
                original_j = self.active_idx if j == 0 else self.inactive_idx
                self.alphas.grad[original_i] += bin_alpha_grads[original_j] * probs[j] * (float(i == j) - probs[i])

        self.old_active_alpha = self.alphas[self.active_idx]
        self.old_inactive_alpha = self.alphas[self.inactive_idx]

    def rescale_alphas(self):
        assert self.old_active_alpha is not None and self.old_inactive_alpha is not None
        a = np.logaddexp(self.alphas[self.active_idx].item(), self.alphas[self.inactive_idx].item())
        b = np.logaddexp(self.old_active_alpha.item(), self.old_inactive_alpha.item())
        self.alphas.data[self.active_idx] -= a - b
        self.alphas.data[self.inactive_idx] -= a - b


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.proj = nn.Conv2d(3, 32, 1)
        self.kernel_1 = canvas.Placeholder()
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.kernel_2 = canvas.Placeholder()

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.kernel_1(x)
        x = self.relu(self.bn(x))
        x = self.kernel_2(x)
        return x


if __name__ == '__main__':
    # Initialize the model
    model = ExampleModel()

    # Sample two kernels
    kernel_pack_1 = canvas.sample(model, example_input=torch.randn(1, 3, 224, 224))
    kernel_pack_2 = canvas.sample(model, example_input=torch.randn(1, 3, 224, 224))

    # Replace the placeholders with the sampled kernels
    model = canvas.replace(model,
                           partial(ParallelKernels, kernel_cls_list=[kernel_pack_1.module, kernel_pack_2.module]),
                           'cpu')
    out = model(torch.randn(1, 3, 224, 224))
    print(model)
