import canvas
import torch
import torchvision


def test_main():
    net = torchvision.models.resnet18()
    canvas.sample(net, torch.zeros(1, 3, 224, 224))
    for _ in range(1000):
        pack = canvas.sample(net)
        net = canvas.replace(net, pack)
        t = net(torch.zeros(1, 3, 224, 224))
