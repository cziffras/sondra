import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .decoder_segformer import SegFormerDecoder, SegFormerSegmentationHead
from .encoder_segformer import SegFormerEncoder

class SegFormerComplex(nn.Module):
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


if __name__ == "__main__":
 
    model = SegFormerComplex(
        in_channels=2,  
        widths=[64, 128, 256, 512],
        depths=[3, 4, 6, 3],
        all_num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        overlap_sizes=[4, 2, 2, 2],
        reduction_ratios=[8, 4, 2, 1],
        mlp_expansions=[4, 4, 4, 4],
        decoder_channels=256,
        scale_factors=[8, 4, 2, 1],
        num_classes=2,  # Pour la segmentation binaire, par exemple
    )
    
    # Création d'un tenseur d'entrée complexe pour un test
    input_tensor = torch.randn((64, 4, 128, 128), dtype=torch.complex64)
    
    # Forward pass
    output = model(input_tensor)
    print(f"Forme de la sortie: {output.shape}")