import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Tensor, Dict

from .decoder_segformer import SegFormerDecoder
from .encoder_segformer import SegFormerEncoder
from .layers_segformer import SegFormerSegmentationHead

class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation


################### Segmentation Segformer Wrapper #############################


class SegmentationSegformer(SegFormer):
    """
    Segmentation wrapper around the SegFormer backbone.

    This class extends the SegFormer model by adding an optional global upsampling 
    layer after the main forward pass. It is configurable via a dictionary `cfg`, 
    which allows control over architectural components such as widths, depths, 
    attention heads, MLP expansions, and decoding behavior.

    Args:
        cfg (Dict): Configuration dictionary for the model.
        input_size (Tuple[int, int]): Input image dimensions (not used directly but available for future use).
        num_classes (int): Number of output classes for segmentation.
    """
    def __init__(self, cfg: Dict, input_size: Tuple[int, int], num_classes: int) -> None:
        num_channels: int = cfg.get("num_channels", 3)
        widths: List[int] = cfg.get("widths", [8, 16, 32, 64])
        depths: List[int] = cfg.get("depths", [3, 4, 6, 3])
        all_num_heads: List[int] = cfg.get("all_num_heads", [1, 2, 4, 8])
        patch_sizes: List[int] = cfg.get("patch_sizes", [7, 3, 3, 3])
        overlap_sizes: List[int] = cfg.get("overlap_sizes", [4, 2, 2, 2])
        reduction_ratios: List[int] = cfg.get("reduction_ratios", [8, 4, 2, 1])
        mlp_expansions: List[int] = cfg.get("mlp_expansions", [4, 4, 4, 4])
        drop_prob: float = cfg.get("drop_prob", 0.1)
        decoder_channels: int = cfg.get("decoder_channels", 64)
        scale_factors: List[int] = cfg.get("scale_factors", [8, 4, 2, 1])
        
        super().__init__(
            in_channels=num_channels,
            widths=widths,
            depths=depths,
            all_num_heads=all_num_heads,
            patch_sizes=patch_sizes,
            overlap_sizes=overlap_sizes,
            reduction_ratios=reduction_ratios,
            mlp_expansions=mlp_expansions,
            decoder_channels=decoder_channels,
            scale_factors=scale_factors,
            num_classes=num_classes,
            drop_prob=drop_prob,
        )

        self.upsample_scale_factor: int = cfg.get("upsample_scale_factor", 1)
        if self.upsample_scale_factor != 1:
            self.upsample_layer = nn.Upsample(
                scale_factor=self.upsample_scale_factor, 
                mode='bilinear', 
                align_corners=False
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the segmentation model.

        It first calls the base SegFormer forward to compute the segmentation map,
        and optionally applies a global upsampling layer if specified in the configuration.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Segmentation map of shape (B, num_classes, H_out, W_out)
        """
        segmentation: Tensor = super().forward(x)
        if hasattr(self, "upsample_layer"):
            segmentation = self.upsample_layer(segmentation)
        return segmentation