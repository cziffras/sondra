import torch
import torch.nn as nn
from torch import Tensor
from torchcvnn.nn.modules import LayerNorm, MultiheadAttention, CGELU, modReLU, BatchNorm2d, Upsample
from torchvision.ops import StochasticDepth
from einops import rearrange
from typing import List, Iterable, Tuple

from ..helpers import chunks 

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2d tensors with complex parameters.

    WARNING : 
    There was an issue with contiguity, apparently solved by using native methods
    instead of einops.
    """
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = LayerNorm(normalized_shape)
        
    def forward(self, x: Tensor) -> Tensor:
        shape_orig = x.shape
        x = x.permute(0, 2, 3, 1)  # b c h w -> b h w c
        x = x.reshape(-1, x.size(-1))  # flat b,h,w
        x = self.ln(x)  # Normalize
        x = x.view(shape_orig[0], shape_orig[2], shape_orig[3], shape_orig[1])  # restore shape b h w c
        x = x.permute(0, 3, 1, 2)  # b h w c -> b c h w
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False,
                dtype=torch.complex64
            ),
            LayerNorm2d(out_channels)
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, 
                channels, 
                kernel_size=reduction_ratio, 
                stride=reduction_ratio,
                dtype=torch.complex64
            ),
            LayerNorm2d(channels),
        )
        self.att = MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        reduced_x = self.reducer(x)
        # reshape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # restore (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=height, w=width)
        return out


class MixMLP(nn.Sequential):
    """MLP with depth-wise convolution for complex tensors"""
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # Couche dense d'entrée
            nn.Conv2d(
                channels, 
                channels, 
                kernel_size=1,
                dtype=torch.complex64           
            ),
            # Convolution depth-wise
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
                dtype=torch.complex64  
            ),
            CGELU(),
            # Couche dense de sortie
            nn.Conv2d(
                channels * expansion, 
                channels, 
                kernel_size=1,
                dtype=torch.complex64
            ),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False, dtype=torch.complex64),
            modReLU(),
            BatchNorm2d(channels)
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1, dtype=torch.complex64)

    def forward(self, features: List[Tensor]) -> Tensor:
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x
