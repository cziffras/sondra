import torch
from torch import Tensor, nn

from ..helpers import chunks
from ..missing_c_nn_layers import StochasticDepth
from .layers_segformer import (
    EfficientMultiHeadAttention,
    LayerNorm2d,
    MixMLP,
    OverlapPatchMerging,
    ResidualAdd,
)


class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = 0.0,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch"),
                )
            ),
        )


class SegFormerEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: list[float],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels,
            out_channels,
            patch_size,
            overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.overlap_patch_merge(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: list[int],
        depths: list[int],
        all_num_heads: list[int],
        patch_sizes: list[int],
        overlap_sizes: list[int],
        reduction_ratios: list[int],
        mlp_expansions: list[int],
        drop_prob: float = 0.0,
    ):
        super().__init__()
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths[:-1]],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions,
                )
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
