import sys

import torch

from .segformer import SegmentationSegformer
from .unet import SegmentationUNet


def build_model(cfg, input_size, num_classes):
    module = sys.modules[__name__]
    model_class = getattr(module, cfg["class"])
    return model_class(cfg, input_size, num_classes)


def load_pretrained_encoder(model, path):
    """
    Copy the encoder of a contrastive checkpoint into a segmentation model.

    Only `model.PRETRAINED_MODULES` are transferred. The pre-training model
    carries a decoder and a head too, but its forward returns before them, so
    copying every matching key would also import untrained weights and report
    them as pre-trained.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    prefixes = tuple(f"{name}." for name in model.PRETRAINED_MODULES)

    encoder = {k: v for k, v in state.items() if k.startswith(prefixes)}
    expected = {k for k in model.state_dict() if k.startswith(prefixes)}

    missing = expected - encoder.keys()
    if missing:
        raise ValueError(
            f"{path} lacks {len(missing)} of the {len(expected)} encoder tensors, "
            f"e.g. {sorted(missing)[0]}: was it produced by the same architecture?"
        )

    model.load_state_dict(encoder, strict=False)
    return len(encoder)
