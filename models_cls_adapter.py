"""Adapt image models to video data by only adapting CLS tokens."""
# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import timm

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
)

class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels):
        super().__init__()

        # [(BT) (1 + HW) D] -> [(BT) (1 + HW) R]
        self.fc1 = nn.Linear(in_channels, adapter_channels)

        # [B T R] -> [B T R]
        # A Transformer to model the interactions between the 
        # CLS tokens of different frames
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=adapter_channels,
            nhead=4,
            dim_feedforward=4 * adapter_channels,
            # dim_feedforward=adapter_channels,
            activation=nn.GELU(),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=1,
        )

        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # Initialize the weights of Transformer such that the output
        # is close to zero: set all weights of the FFN to zero
        nn.init.constant_(self.fc1.weight, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        nn.init.constant_(self.transformer.layers[0].self_attn.out_proj.weight, 0.)
        nn.init.constant_(self.transformer.layers[0].self_attn.out_proj.bias, 0.)
        nn.init.constant_(self.transformer.layers[0].linear1.weight, 0.)
        nn.init.constant_(self.transformer.layers[0].linear1.bias, 0.)
        nn.init.constant_(self.transformer.layers[0].linear2.weight, 0.)
        nn.init.constant_(self.transformer.layers[0].linear2.bias, 0.)


    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        x_id = x

        # Only pick the CLS token: (BT) (1 + HW) D -> (BT) D -> B T D
        x = x[:, 0, :]
        x = einops.rearrange(x, "(B T) D -> B T D", B=B)

        # Down projection: B T D -> B T R
        x = self.fc1(x)

        # Transformer: B T R -> B T R
        x = self.transformer(x)

        # Up projection: B T R -> B T D
        x = self.fc2(x)

        # Reshape: B T D -> (BT) D
        x = einops.rearrange(x, "B T D -> (B T) D", B=B)

        # Residual connection
        x_id[:, 0, :] += x
        return x_id


if __name__ == "__main__":
    import shared.utils as su

    adapter = Adapter(768, 128)
    su.misc.num_params(adapter)

    x = torch.randn(32, 1 + 196, 768)
    y = adapter(x, 16)
    print(x.shape, y.shape)
    assert (x == y).all()
    print("Test for Adapter passed.")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                #  adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            # kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor:
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                #  adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                # adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                #  adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads,
            adapter_width, adapter_layers,
            # adapter_kernel_size,
            adapter_pre_attn, adapter_pre_mlp
        )

        self.ln_post = LayerNorm(width)

        for n, p in self.named_parameters():
          if 'adapter' not in n:
            p.requires_grad_(False)
            # p.data = p.data.half()
            p.data = p.data.float()
        
        self.dropout = nn.Dropout(0.5)
        if num_classes > 0:
            self.fc = nn.Linear(width, num_classes)
            nn.init.normal_(self.fc.weight, std=0.02)
            nn.init.constant_(self.fc.bias, 0.)
        else:
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        x = x[:, :, 0, :].mean(dim=1)

        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def clip_vit_base_patch16_cls_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        # adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_base_patch16_cls_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        # adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model


class CLIP4Clip(nn.Module):
    """
    CLIP baseline with mean pooling of frame features.
    """
    def __init__(self, model_id="vit_base_patch16_clip_224.openai", num_classes=174):
        super().__init__()
        self.model = timm.create_model(model_id, pretrained=True)
        self.model.head = nn.Identity()

        # Classifier
        if num_classes > 0:
            self.fc = nn.Linear(self.model.num_features, num_classes)
            nn.init.normal_(self.fc.weight, std=0.02)
            nn.init.constant_(self.fc.bias, 0.)
        else:
            self.fc = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, T, C, H, W)
        Returns:
            torch.Tensor: (B, num_classes)
        """
        B = len(x)
        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        z = self.model(x)
        z = einops.rearrange(z, "(B T) C -> B T C", B=B)
        y = self.fc(z.mean(dim=1))
        return y
    

def clip4clip_vit_base_patch16_meanpool(num_classes=174):
    model = CLIP4Clip(
        num_classes=num_classes,
        model_id="vit_base_patch16_clip_224.openai",
    )
    return model


if __name__ == "__main__":
    backbone = clip_vit_base_patch16_cls_adapter12x384(num_classes=0)
    su.misc.num_params(backbone)
    su.misc.num_trainable_params(backbone)

    B, T, C, H, W = 4, 16, 3, 224, 224
    x = torch.randn(B, C, T, H, W)
    y = backbone(x)
    print(x.shape, y.shape)

    import timm
    model = CLIP4Clip()
    su.misc.num_params(model)
    y = model(x)
    print(x.shape, y.shape)
