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
        self.alphas = nn.Parameter(torch.randn(len(kernel_cls_list)))

    def forward(self, x):
        softmax_alphas = F.softmax(self.alphas, dim=0)
        stacked_outs = torch.stack([kernel(x) for kernel in self.module_list], dim=0)
        return torch.einsum('i,i...->...', softmax_alphas, stacked_outs)


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
