import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import canvas


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.block = canvas.Placeholder(dim)
        layer_scale_init_value = 1e-2
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, )), requires_grad=True)

    def forward(self, x):
        return x + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm(x))


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


# noinspection PyDefaultArgument
class Former(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], drop_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if not flag:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i]) for _ in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(canvas.init_weights)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# noinspection PyUnusedLocal
@register_model
def canvas_former(pretrained=False, **kwargs):
    # noinspection PyTypeChecker
    model = Former(
        embed_dims=[32, 64, 160, 256],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 5, 5],
        **kwargs)
    model.default_cfg = _cfg()
    return model
