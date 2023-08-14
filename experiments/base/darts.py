import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict
import canvas
from . import log


def get_final_model(model, wsharing):
    kernel_dict = {}
    # Helper function to recursively search for conv layers
    def find(module):
        for name, child in module.named_children():
            if isinstance(child, ParallelPlaceholder):
                kernel_dict[name] = child
            elif isinstance(child, nn.Module):
                find(child)
                
    find(model)
    for name, ensembled_kernel in kernel_dict.items():
        final_kernel = ensembled_kernel.get_max_weight_kernel()
        setattr(model, name, final_kernel)
            
            
class ParallelPlaceholder(nn.Module):
    """
    A module representing a parallel combination of multiple kernels with weighted outputs.

    Args:
        kernel_cls_list (list): List of kernel class types to be instantiated.
        i (int): Index of the placeholder.
        **kwargs: Keyword arguments passed to kernel constructors(c, h, w).

    Attributes:
        i (int): Rank of this placeholder in the model.
        module_list (nn.ModuleList): List of instantiated kernel modules.
        alphas (nn.Parameter): Learnable architecture parameter representing weights of kernels.

    Methods:
        forward(x): Forward pass through the module.
        get_max_weight_kernel(): Get the kernel with the maximum weight.
        print_parameters(j): Print the parameters of the module when it's trained.
        get_alphas(): Get the alpha values.

    """
    def __init__(self, kernel_cls_list, i, **kwargs):
        super().__init__()
        self.i = i
        assert len(kernel_cls_list) >= 1
        self.module_list = nn.ModuleList([kernel_cls(*kwargs.values()) for kernel_cls in kernel_cls_list])
        self.alphas = nn.Parameter((1e-3) * torch.randn(len(kernel_cls_list)))

    def forward(self, x: torch.Tensor):
        softmax_alphas = F.softmax(self.alphas, dim=0)
        stacked_outs = torch.stack([kernel(x) for kernel in self.module_list], dim=0)
        return torch.einsum('i,i...->...', softmax_alphas, stacked_outs)

    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.alphas)
        return self.module_list[max_weight_idx]
    
    def print_parameters(self, j, parameters):
        logger = log.get_logger()
        
        # Print parameters in each placeholder  
        logger.info(f'In {self.i}th Placeholder')  
        
        # Alpha
        logger.info(f'####### ALPHA After {j}th epoch #######')
        logger.info('# Alphas')
        logger.info(F.softmax(self.alphas, dim=0))
        logger.info('#####################')
    
    
def get_alphas(model):
    return nn.ParameterList(placeholder.canvas_placeholder_kernel.alphas for placeholder in model.canvas_cached_placeholders)


def get_weights(model):
    weights = nn.ParameterList(param for name, param in model.named_parameters() if 'alphas' not in name)
    return weights
    
    
class InGtOut(nn.Module):
    """
    A custom module that applies when Input channels greater than Output channels

    Args:
        factor (int): Split factor for the input tensor.
    """
    def __init__(self, factor):
        super(InGtOut, self).__init__()
        self.factor = factor 
        self.layer = canvas.Placeholder()
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Aggregated output tensor.
        """
        split_outputs = []
        # print(f"old channel = {print(x.shape)}, factor = {self.factor}")
        tensors = torch.split(x, x.shape[1] // self.factor, dim = 1)
        for tensor in tensors:
            output = self.layer(tensor)
            split_outputs.append(output)
        aggregated_output = torch.sum(torch.stack(split_outputs), dim=0)
        return aggregated_output
    
    
class OutGtIn(nn.Module):
    """
    A custom module that applies when Output channels greater than Input channels

    Args:
        factor (int): Split factor for the input tensor.
    """
    def __init__(self, factor):
        super(OutGtIn, self).__init__()
        self.factor = factor
        self.layer = canvas.Placeholder()
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated output tensor.
        """
        output = self.layer(x) 
        concatenated_tensor = torch.cat([output for _ in range(self.factor)], dim=1)
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
            if module.in_channels * module.out_channels < 256 * 256 - 1:
                return False
            return True
        case "resblock":
            return True
        
        
def replace_module_with_placeholder(module: nn.Module, old_module_types: Dict[nn.Module, str], filter: Callable = filter):
    if isinstance(module, canvas.Placeholder):
        return 0, 1
    # assert old_module_type == nn.Conv2d or old_module_type == SqueezeExcitation
    replaced, not_replaced = 0, 0
    for name, child in module.named_children():
        if type(child) in old_module_types:
            string_name = old_module_types[type(child)]
            match string_name:
                case "conv":
                    if filter(child, string_name):
                        replaced += 1          
                        if (child.in_channels == child.out_channels):
                            setattr(module, name, canvas.Placeholder())
                            
                        elif (child.in_channels < child.out_channels):
                            factor = child.out_channels // child.in_channels
                            setattr(module, name, OutGtIn(factor))
                        else:
                            factor = child.in_channels // child.out_channels
                            setattr(module, name, InGtOut(factor))
                    else:
                        not_replaced += 1
                case "resblock":
                    if filter(child, string_name):
                        replaced += 1
                        if (child.downsample is None):
                            setattr(module, name, canvas.Placeholder())
                        else:
                            factor = 2
                            setattr(module, name, OutGtIn(factor))
                    else:
                        not_replaced += 1           
        elif len(list(child.named_children())) > 0:
            count = replace_module_with_placeholder(child, old_module_types, filter)
            replaced += count[0]
            not_replaced += count[1]
    return replaced, not_replaced


""" Architect controls architecture of cell by computing gradients of alphas 
    NAS训练算法中的第1步: 更新架构参数 α
    根据论文可知 dα Lval(w*, α) 约等于 dα Lval(w', α)    w' = w - ξ * dw Ltrain(w, α)
"""
import copy


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay, criterion):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net                      # network
        self.v_net = copy.deepcopy(net)     # 不直接用外面的optimizer来进行w的更新，而是自己新建一个network，主要是因为我们这里的更新不能对Network的w进行更新
        self.criterion = criterion
        self.w_momentum = w_momentum
        
        self.w_weight_decay = w_weight_decay    # 正则化项用来防止过拟合
    def get_loss(self, model, x, y):
        logits = model.forward(x)
        return self.criterion(logits, y)
    def virtual_step(self, trn_X, trn_y, xi, w_optim, criterion):
        """
        Compute unrolled weight w' (virtual step)

        根据公式计算 w' = w - ξ * dw Ltrain(w, α)   
        Monmentum公式：  dw Ltrain -> v * w_momentum + dw Ltrain + w_weight_decay * w 
        -> m + g + 正则项
  
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)  即公式中的 ξ
            w_optim: weights optimizer 用来更新 w 的优化器
        """
        # forward & calc loss
        # loss = self.net.loss(trn_X, trn_y) # L_trn(w)
        loss = self.get_loss(self.net, trn_X, trn_y)
        # compute gradient 计算  dw L_trn(w) = g
        gradients = torch.autograd.grad(loss, get_weights(self.net))

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(get_weights(self.net), get_weights(self.v_net), gradients):
                # m = v * w_momentum  用的就是Network进行w更新的momentum
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum

                # 做一步momentum梯度下降后更新得到 w' = w - ξ * (m + dw Ltrain(w, α) + 正则项 )
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))   

            # synchronize alphas 更新了v_net的alpha
            for a, va in zip(get_alphas(self.net), get_alphas(self.v_net)):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        计算目标函数关于 α 的近似梯度
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim, self.criterion)

        # calc unrolled loss
        # loss = self.v_net.loss(val_X, val_y) # L_val(w', α)  在使用w', 新alpha的net上计算损失值
        loss = self.get_loss(self.v_net, trn_X, trn_y)
        # compute gradient
        v_alphas = tuple(get_alphas(self.v_net))
        v_weights = tuple(get_weights(self.v_net))
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]    # dα L_val(w', α)   梯度近似后公式第一项
        dw = v_grads[len(v_alphas):]        # dw' L_val(w', α)  梯度近似后公式第二项的第二个乘数

        hessian = self.compute_hessian(dw, trn_X, trn_y)        # 梯度近似后公式第二项

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(get_alphas(self.net), dalpha, hessian):
                alpha.grad = da - xi*h    # 求出了目标函数的近似梯度值

    def compute_hessian(self, dw, trn_X, trn_y):   
        """
        求经过泰勒展开后的第二项的近似值
        dw = dw` { L_val(w`, alpha) }  输入里已经给了所有预测数据的dw
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)    [1]
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()   # 把每个 w 先拉成一行，然后把所有的 w 摞起来，变成 n 行, 然后求L2值
        eps = 0.01 / norm

        # w+ = w + eps * dw`
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p += eps * d        # 将model中所有的w'更新成 w+
        loss = self.get_loss(self.net, trn_X, trn_y)      # L_trn(w+)
        dalpha_pos = torch.autograd.grad(loss, get_alphas(self.net)) # dalpha { L_trn(w+) }

        # w- = w - eps * dw`
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p -= 2. * eps * d   # 将model中所有的w'更新成 w-,   w- = w - eps * dw = w+ - eps * dw * 2, 现在的 p 是 w+
        loss = self.get_loss(self.net, trn_X, trn_y)      # L_trn(w-)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p += eps * d        # 将模型的参数从 w- 恢复成 w,  w = w- + eps * dw

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]  # 利用公式 [1] 计算泰勒展开后第二项的近似值返回
        return hessian
