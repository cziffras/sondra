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

    Every learnt tensor of the encoder must be in the checkpoint, its running
    statistics need not be: see `_from_layernorm2d`.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    prefixes = tuple(f"{name}." for name in model.PRETRAINED_MODULES)

    encoder = _from_layernorm2d(
        {k: v for k, v in state.items() if k.startswith(prefixes)}, model.state_dict()
    )
    # the learnt tensors must all be there, the running statistics may not
    expected = {k for k, _ in model.named_parameters() if k.startswith(prefixes)}

    missing = expected - encoder.keys()
    if missing:
        raise ValueError(
            f"{path} lacks {len(missing)} of the {len(expected)} encoder tensors, "
            f"e.g. {sorted(missing)[0]}: was it produced by the same architecture?"
        )

    model.load_state_dict(encoder, strict=False)
    return len(encoder)


def _from_layernorm2d(state, current):
    """
    Reads the checkpoints of the pre-trainings run while the SegFormer normalised with
    `LayerNorm2d`, before it moved to BatchNorm2d.

    Already documented see the docstring of `LayerNorm2d` : in training, the torchcvnn LayerNorm 
    computes exactly what BatchNorm2d computes which obviously is an error. Pretraining is ran 
    locally, instead of rerunning from scratch we note that due to this implementation error in
    `torchcvnn` these encoders are the ones a BatchNorm2d model would have learnt. To verify so 
    we ran several steps deterministically with either the erroneous `LayerNorm2d` or `BatchNorm2d`
    plugged loss differences were mostly float32 roundoff, zero so to say.
    
    Only their layout differs: the parameters sit under an extra `ln.` level, and there
    are no running statistics, which the fine-tuning estimates since it runs in training
    mode.
    """
    renamed = {key: key.replace(".ln.", ".") for key in state}
    return {
        (renamed[key] if renamed[key] in current else key): value for key, value in state.items()
    }
