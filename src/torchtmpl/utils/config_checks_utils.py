import math
import itertools
import torch
import yaml

def check_model_params_validity(config, use_cuda, contrastive): 

    from ..data import get_dataloaders
    from ..models import build_model

    data_config = config["data"]

    loader, _, input_size, num_classes = get_dataloaders(
        data_config, use_cuda, contrastive
    )

    first_batch_inputs, _ = next(iter(loader))

    model = build_model(
        config["model"],
        input_size,
        num_classes
    )

    try:
        _ = model(first_batch_inputs)
    except Exception as err: 
        raise ValueError(f"Invalid forward : check model's config. For more details : {err}") from err
    
    print("############## Config is correct ##############")


    