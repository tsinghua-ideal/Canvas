import numpy as np
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F
import canvas


class ParallelKernels(nn.Module):
    def __init__(self, kernel_cls_list, **kwargs):
        super().__init__()
        assert len(kernel_cls_list) > 1
        self.module_list = nn.ModuleList([kernel_cls(*kwargs.values()) for kernel_cls in kernel_cls_list])
        self.kernel_alphas = nn.Parameter(torch.full((len(self), ), 0.05))
        self.kernel_binary_gate = nn.Parameter(torch.Tensor(len(self)))
        self.active_only = None
        self.module_list_backup = None
        self.active_idx, self.inactive_idx = None, None
        self.old_active_inactive_alpha = None

    def get_softmaxed_kernel_alphas(self):
        return F.softmax(self.kernel_alphas, dim=0)
    
    def set_indices(self, active_idx, inactive_idx, active_only):
        self.active_only = active_only
        self.active_idx, self.inactive_idx = active_idx, inactive_idx
        self.kernel_binary_gate.data.zero_()
        self.kernel_binary_gate.data[active_idx] = 1
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
        active_module = self.module_list[self.active_idx]
        if self.active_only:
            return active_module(x)
        else:
            inactive_module = self.module_list[self.inactive_idx]
            active = self.kernel_binary_gate[self.active_idx] * active_module(x)
            inactive = self.kernel_binary_gate[self.inactive_idx] * inactive_module(x).detach()
            return active + inactive

    def set_alpha_grad(self):
        bin_alpha_grads = self.kernel_binary_gate.grad.detach()
        self.kernel_alphas.grad = torch.zeros_like(self.kernel_alphas)
        probs = F.softmax(torch.stack([self.kernel_alphas[self.active_idx], self.kernel_alphas[self.inactive_idx]]), dim=0)
        for i in range(2):
            for j in range(2):
                original_i = self.active_idx if i == 0 else self.inactive_idx
                original_j = self.active_idx if j == 0 else self.inactive_idx
                self.kernel_alphas.grad.data[original_i] += bin_alpha_grads[original_j] * probs[j] * (float(i == j) - probs[i])
        self.old_active_inactive_alpha = torch.tensor([self.kernel_alphas[self.active_idx], self.kernel_alphas[self.inactive_idx]], device='cuda')
    
    def rescale_kernel_alphas(self):
        assert self.old_active_inactive_alpha[0] is not None and self.old_active_inactive_alpha[1] is not None
        a = torch.logaddexp(self.kernel_alphas[self.active_idx], self.kernel_alphas[self.inactive_idx])
        b = torch.logaddexp(self.old_active_inactive_alpha[0], self.old_active_inactive_alpha[1])
        self.kernel_alphas.data[self.active_idx] -= a - b
        self.kernel_alphas.data[self.inactive_idx] -= a - b
    

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


