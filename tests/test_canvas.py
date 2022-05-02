import canvas
import torch
import torchvision


def test_seed():
    net = torchvision.models.resnet18()
    canvas.seed(1998)
    pack_1 = canvas.sample(net, torch.zeros(1, 3, 224, 224))
    canvas.seed(1998)
    pack_2 = canvas.sample(net, torch.zeros(1, 3, 224, 224))
    assert pack_1.code == pack_2.code
    assert pack_1.graphviz == pack_2.graphviz


def test_sample():
    net = torchvision.models.resnet18()
    canvas.sample(net, torch.zeros(1, 3, 224, 224))
    for _ in range(1000):
        pack = canvas.sample(net)
        net = canvas.replace(net, pack)
        t = net(torch.zeros(1, 3, 224, 224))
