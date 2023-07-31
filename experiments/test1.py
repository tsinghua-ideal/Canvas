
import torch
import math
from base import dataset, device, models, parser, trainer, log
import torch.nn as nn
from canvas import Placeholder
import canvas
from typing import Union, Callable, List, Dict
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # 下面的全连接层的输入维度取决于你的输入数据以及你的卷积和池化层的参数
        self.fc1 = nn.Linear(16 * 7 * 7, 120) # 假设在卷积层后，图像的尺寸为7x7
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        
        # flatten
        print(f"x.shape after conv:{x.shape}")
        x = x.view(-1, self.num_flat_features(x))
        print(f"x.shape after flatten:{x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

       
class ResidualBlock_low(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock_low, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet_18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
 
    def forward(self, x):#3*32*32
        out = self.conv1(x)#64*32*32
        out = self.layer1(out)#64*32*32
        out = self.layer2(out)#128*16*16
        out = self.layer3(out)#256*8*8
        out = self.layer4(out)#512*4*4
        out = F.avg_pool2d(out, 4)#512*1*1
        out = out.view(out.size(0), -1)#512
        out = self.fc(out)
        return out
def ResNet18():
    return ResNet_18(ResidualBlock_low)
class in_gt_out(nn.Module):
    def __init__(self, factor):
        super(in_gt_out, self).__init__()
        self.factor = factor 
        self.layer = canvas.Placeholder()
    def forward(self, x):
        split_outputs = []
        print(f"old channel = {print(x.shape)}, factor = {self.factor}")
        tensors = torch.split(x, x.shape[1] // self.factor, dim = 1)
        for tensor in tensors:
            output = self.layer(tensor)
            split_outputs.append(output)
        output = torch.sum(torch.stack(split_outputs), dim=0)
        return output
class out_gt_in(nn.Module):
    def __init__(self, factor):
        super(out_gt_in, self).__init__()
        self.factor = factor
        self.layer = canvas.Placeholder()
    def forward(self, x):
        outputs = []
        for i in range(self.factor):
            outputs.append(self.layer(x))
        concatenated_tensor = torch.cat(outputs, dim=1)
        print(f"old channel = {x.shape[1]}, new channel = {concatenated_tensor.shape[1]}")
        return concatenated_tensor
    
def filter(module: nn.Module, type: str = "conv", max_count: int = 0) -> bool:
    match type:
        case "conv":
            if module.groups > 1:
                return False
            if module.kernel_size not in [(1, 1), (3, 3), (5, 5), (7, 7)]:
                return False
            if module.kernel_size == (1, 1) and module.padding != (0, 0):
                return False
            if module.kernel_size == (3, 3) and module.padding != (1, 1):
                return False
            if module.kernel_size == (5, 5) and module.padding != (2, 2):
                return False
            if module.kernel_size == (7, 7) and module.padding != (3, 3):
                return False
            width = math.gcd(module.in_channels, module.out_channels)
            if width != min(module.in_channels, module.out_channels):
                return False
            count = max(module.in_channels, module.out_channels) // width
            if count > max_count != 0:
                return False
            return True
        case "resblock":
            return True
def replace_module_with_placeholder(module: nn.Module, old_module_types: Dict[nn.Module, str], filter: Callable = filter):
    if isinstance(module, Placeholder):
            return 0, 1
    # assert old_module_type == nn.Conv2d or old_module_type == SqueezeExcitation
    
    replaced, not_replaced = 0, 0
    for name, child in module.named_children():
        if type(child) in old_module_types:
            print("gotcha")
            string_name = old_module_types[type(child)]
            match string_name:
                case "conv":
                    if filter(child, string_name):
                        replaced += 1
                        
                        print(f"cin = {child.in_channels}, cout = {child.out_channels}")
                        if (child.in_channels > child.out_channels):
                            factor = child.in_channels // child.out_channels
                            print(f"factor = {factor}")
                            setattr(module, name,
                                    in_gt_out(factor
                                                    ))
                        elif (child.in_channels <= child.out_channels):
                            factor = child.out_channels // child.in_channels
                            print(f"factor = {factor}")
                            setattr(module, name,
                                    out_gt_in(factor
                                                    ))
                    else:
                        not_replaced += 1
                case "resblock":
                    if filter(child, string_name):
                        replaced += 1
                        setattr(module, name,
                                Placeholder(
                                                ))
                    else:
                        not_replaced += 1       
                                 
        elif len(list(child.named_children())) > 0:
            count = replace_module_with_placeholder(child, old_module_types, filter)
            replaced += count[0]
            not_replaced += count[1]
    return replaced, not_replaced
    
def test_model(model: nn.Module, module_dict):
    # output = model(input_tensor)
    print("Model (before replacement):")
    print(model)
    replaced, not_replaced = replace_module_with_placeholder(model, module_dict)
    # output = model(input_tensor)
    print("Model  (after replacement):")
    print(model)
    print(f"replaced kernel = {replaced}, not replaced kernel = {not_replaced}")
    kernel_pack = canvas.sample(model, example_input=torch.randn(1, 3, 224, 224))
    # Replace the original kernel with the sampled one
    canvas.replace(model, kernel_pack.module)
    print(model)

if __name__ == '__main__':

# 加载预训练的 ResNet-18 模型
    # model = Net()
    model = models.resnet18(pretrained=True)
    # module_dict = {
    #     nn.Conv2d: "conv",
    #     ResidualBlock_low: "resblock"
    # }
    module_dict = {
        nn.Conv2d: "conv"
    }    
    test_model(model, module_dict)
