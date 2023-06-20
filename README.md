## Canvas: End-to-End Kernel Architecture Search in Neural Networks

This is the development repository of Canvas, a library for sampling fine-grained PyTorch kernels (similar to NAS). Canvas samples from a rich set of fine-grained primitives to stochastically and iteratively construct new kernels and evaluate them according to user-specified constraints. Canvas supports freely adjustable tensor dimension sizes inside the kernel and uses two levels of solvers to satisfy structural legality and fully utilize model budgets.

### Quick Start

#### Installation

```bash
python setup.py install
```

#### Interfaces

Canvas only has 3 interfaces:
- `canvas.Placeholder()`: A placeholder module that can be used to replace any module in a neural network, you can simply declare any placeholder module and use it as a normal module in your neural network. Note that the placeholder module will produce a same-shape output as the input. The shape of the input size should be [N, C\*, H\*, W\*]. "\*" means that the dimension can be non-existent.
- `canvas.sample()`: Sample an available kernel for a module from the search space. This function will find all placeholders in the module, and sample an available to substitute the originals.
- `canvas.replace()`: Replace all kernel placeholders of n with sample kernels in pack.

For more details, please refer to the doc-string of the Python interfaces.

#### Example

Below is a simple example of using Canvas to search for a kernel in a convolutional neural network.

```python
import canvas
import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.proj = nn.Conv2d(3, 32, 1)
        # Initialize the module to be sampled with `canvas.Placeholder()`
        self.kernel_1 = canvas.Placeholder()
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # Initialize the module to be sampled with `canvas.Placeholder()`
        self.kernel_2 = canvas.Placeholder()

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        # All placeholders will produce a same-shape output as the input
        x = self.kernel_1(x)
        x = self.relu(self.bn(x))
        x = self.kernel_2(x)
        return x

if __name__ == "__main__":
    # Initialize the model
    model = ExampleModel()
    # Sample a kernel
    # You may also repeat the sampling process in a loop to find a better kernel
    kernel_pack = canvas.sample(model, example_input=torch.randn(1, 3, 224, 224))
    # Replace the original kernel with the sampled one
    canvas.replace(model, kernel_pack.module)
    # Print PyTorch implementation of the sampled kernel
    print(f"Sampled kernel code: {kernel_pack.torch_code}")
```

An example output of the PyTorch implementation of the sampled kernel is shown below:
```python
class Kernel_4740052357514212317(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4740052357514212317, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K5_D2: p_1
        pass
        # BMM_0_1: p_2
        pass
        # ReLU: p_3
        pass
        # UnfoldH_K5_D1: p_4
        pass
        # GeLU: p_5
        pass
        # Scale_0/1/C_1/1/C_1/3/KW: p_6
        self.p_6_w = nn.Parameter(torch.ones((1, self.c, self.c, 5,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.02)
        # BMM_0_0: p_7
        pass
        # Output: p_8
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K5_D2: p_1
        t_1 = F.unfold(t_0, (1, 5), dilation=(1, 2), padding=(0, 4))
        t_1 = t_1.view(self.n, self.c, 5, self.h, self.w)
        # BMM_0_1: p_2
        t_2_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_2_rhs = t_1.view(self.n, self.c * 5, self.h * self.w).transpose(1, 2)        
        t_2 = torch.bmm(t_2_lhs, t_2_rhs) / math.sqrt(self.h * self.w)
        t_2 = t_2.view(self.n, self.c, self.c, 5)
        # ReLU: p_3
        t_3 = torch.relu(t_0)
        # UnfoldH_K5_D1: p_4
        t_4 = F.unfold(t_3, (5, 1), dilation=(1, 1), padding=(2, 0))
        t_4 = t_4.view(self.n, self.c, 5, self.h, self.w)
        # GeLU: p_5
        t_5 = F.gelu(t_4)
        # Scale_0/1/C_1/1/C_1/3/KW: p_6
        t_6 = self.p_6_w * t_2
        # BMM_0_0: p_7
        t_7_lhs = t_6.view(self.n, self.c, self.c * 5)        
        t_7_rhs = t_5.view(self.n, self.c * 5, self.h * self.w)        
        t_7 = torch.bmm(t_7_lhs, t_7_rhs) / math.sqrt(self.c * 5)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Output: p_8
        return t_7.view(self.n, self.c, self.h, self.w)
```

### Citation

```bibtex
@misc{zhao2023canvas,
      title={Canvas: End-to-End Kernel Architecture Search in Neural Networks}, 
      author={Chenggang Zhao and Genghan Zhang and Mingyu Gao},
      year={2023},
      eprint={2304.07741},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
