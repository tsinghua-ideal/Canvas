import canvas
import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.proj = nn.Conv2d(3, 32, 1)
        self.kernel_1 = canvas.Placeholder(32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.kernel_2 = canvas.Placeholder(32)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.kernel_1(x)
        x = self.relu(self.bn(x))
        x = self.kernel_2(x)
        return x


def demo():
    net = ExampleModel()
    pack = canvas.sample(net,
                         example_input=torch.zeros(1, 3, 32, 32),
                         allow_dynamic=True)
    net = canvas.replace(net, pack)
    # Do something with your new net ...!


def test_seed():
    net = ExampleModel()
    canvas.seed(1998)
    pack_1 = canvas.sample(net, torch.zeros(1, 3, 32, 32))
    canvas.seed(1998)
    pack_2 = canvas.sample(net, torch.zeros(1, 3, 32, 32))
    assert pack_1.code == pack_2.code
    assert pack_1.graphviz == pack_2.graphviz


def test_sample():
    net = ExampleModel()
    canvas.sample(net, torch.zeros(1, 3, 32, 32))
    for _ in range(10):
        pack = canvas.sample(net)
        print(pack.code)
        net = canvas.replace(net, pack)
        t = net(torch.zeros(1, 3, 32, 32))
