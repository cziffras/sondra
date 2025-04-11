import math
import itertools
import torch
import yaml

from .training_utils import one_forward


def check_model_params_validity(config, use_cuda, contrastive): 

    from ..data import get_dataloaders
    from ..models import build_model

    data_config = config["data"]

    loader, _, input_size, num_classes = get_dataloaders(
        data_config, use_cuda, contrastive
    )

    first_batch = next(iter(loader))

    model = build_model(
        config["model"],
        input_size,
        num_classes
    )

    try:
        _, _ = one_forward(model, first_batch, 'cuda' if use_cuda else 'cpu')
    except Exception as err: 
        raise ValueError(f"Invalid forward : check model's config. For more details : {err}") from err


    