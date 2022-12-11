import canvas
import torch
from torch import nn


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


class ExampleModelSingleSpatial(nn.Module):
    def __init__(self):
        super(ExampleModelSingleSpatial, self).__init__()
        self.proj = nn.Conv1d(3, 32, 1)
        self.kernel_1 = canvas.Placeholder()
        self.relu = nn.ReLU(inplace=True)
        self.kernel_2 = canvas.Placeholder()

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.kernel_1(x)
        x = self.relu(x)
        x = self.kernel_2(x)
        return x


class ExampleModelNoneSpatial(nn.Module):
    def __init__(self):
        super(ExampleModelNoneSpatial, self).__init__()
        self.proj = nn.Linear(3, 32)
        self.kernel_1 = canvas.Placeholder()
        self.relu = nn.ReLU(inplace=True)
        self.kernel_2 = canvas.Placeholder()

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.kernel_1(x)
        x = self.relu(x)
        x = self.kernel_2(x)
        return x


def demo():
    net = ExampleModel()
    pack = canvas.sample(net, example_input=torch.zeros(1, 3, 32, 32))
    net = canvas.replace(net, pack.module, 'cpu')
    # Do something with your new net ...!


def test_seed():
    net = ExampleModel()
    canvas.seed(1998)
    pack_1 = canvas.sample(net, torch.zeros(1, 3, 32, 32))
    canvas.seed(1998)
    pack_2 = canvas.sample(net, torch.zeros(1, 3, 32, 32))
    assert pack_1.torch_code == pack_2.torch_code
    assert pack_1.graphviz_code == pack_2.graphviz_code


def test_different_spatial():
    canvas.seed(1998)
    for model_cls, shape in [
        (ExampleModel, (1, 3, 32, 32)),
        (ExampleModelSingleSpatial, (1, 3, 32)),
        (ExampleModelNoneSpatial, (1, 3))
    ]:
        net = model_cls()
        pack = canvas.sample(net, torch.zeros(shape))
        net = canvas.replace(net, pack.module, 'cpu')
        t = net(torch.zeros(shape))


def test_sample():
    net = ExampleModel()
    canvas.sample(net, torch.zeros(1, 3, 32, 32))
    for _ in range(10):
        pack = canvas.sample(net)
        net = canvas.replace(net, pack.module, 'cpu')
        t = net(torch.zeros(1, 3, 32, 32))


def test_empty_sample():
    for _ in range(10):
        pack = canvas.empty_sample()


def test_kernel_pack_hash():
    # Debug sampling produces the same kernel.
    canvas.debug_sample()
