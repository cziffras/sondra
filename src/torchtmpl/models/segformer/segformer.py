import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchcvnn.nn.modules import Upsample, modReLU

from .decoder_segformer import SegFormerDecoder
from .encoder_segformer import SegFormerEncoder
from .layers_segformer import LayerNorm2d, SegFormerSegmentationHead


class SegFormer(nn.Module):
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
        decoder_channels: int,
        scale_factors: list[int],
        num_classes: int,
        drop_prob: float = 0.0,
        contrastive: bool = False,
        proj_dim: int = 128,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.contrastive = contrastive
        self.dtype = dtype

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

        if self.contrastive:
            self.proj_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(c, proj_dim, kernel_size=1, bias=False, dtype=self.dtype),
                        LayerNorm2d(proj_dim),
                        modReLU(),
                    )
                    for c in widths
                ]
            )

    def forward(self, x: Tensor):

        x = x.to(self.dtype)
        feats = self.encoder(x)

        if self.contrastive:
            emb_list = []
            for feat, head in zip(feats, self.proj_heads):
                z = head(feat)
                z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
                norm = z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
                z = z / norm

                emb_list.append(z)
            return emb_list

        x_dec = self.decoder(feats[::-1])
        seg = self.head(x_dec)

        return seg


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

    def __init__(self, cfg: dict, input_size: tuple[int, int], num_classes: int) -> None:
        num_channels: int = cfg.get("num_channels", 3)
        widths: list[int] = cfg.get("widths", [8, 16, 32, 64])
        depths: list[int] = cfg.get("depths", [3, 4, 6, 3])
        all_num_heads: list[int] = cfg.get("all_num_heads", [1, 2, 4, 8])
        patch_sizes: list[int] = cfg.get("patch_sizes", [7, 3, 3, 3])
        overlap_sizes: list[int] = cfg.get("overlap_sizes", [4, 2, 2, 2])
        reduction_ratios: list[int] = cfg.get("reduction_ratios", [8, 4, 2, 1])
        mlp_expansions: list[int] = cfg.get("mlp_expansions", [4, 4, 4, 4])
        drop_prob: float = cfg.get("drop_prob", 0.1)
        decoder_channels: int = cfg.get("decoder_channels", 64)
        scale_factors: list[int] = cfg.get("scale_factors", [8, 4, 2, 1])
        contrastive: bool = cfg.get("contrastive", False)

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
            contrastive=contrastive,
        )

        self.upsample_scale_factor: int = cfg.get("upsample_scale_factor", 1)
        if self.upsample_scale_factor != 1:
            self.upsample_layer = Upsample(
                scale_factor=self.upsample_scale_factor, mode="bilinear", align_corners=False
            )

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        """
        Forward pass through the segmentation model.

        In supervised mode this returns the segmentation map of shape
        (B, num_classes, H_out, W_out), optionally upsampled by
        `upsample_scale_factor`. In contrastive mode the backbone returns one
        projected embedding per stage instead, and no upsampling applies.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
        """
        out = super().forward(x)
        if self.contrastive:
            return out
        if hasattr(self, "upsample_layer"):
            out = self.upsample_layer(out)
        return out
