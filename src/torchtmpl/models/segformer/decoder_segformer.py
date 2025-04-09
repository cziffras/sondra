import torch
import torch.nn as nn
from torch import Tensor
from torchcvnn.nn.modules import modReLU, BatchNorm2d, Upsample
from torchvision.ops import StochasticDepth
from einops import rearrange
from typing import List

from .helpers import chunks 



class SegFormerDecoderBlock(nn.Sequential):
    """Bloc décodeur pour le SegFormer à tenseurs complexes."""
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64),
        )


class SegFormerDecoder(nn.Module):
    """Décodeur complet pour le SegFormer à tenseurs complexes."""
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    """Tête de segmentation pour le SegFormer à tenseurs complexes."""
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
