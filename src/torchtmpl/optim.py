import torch
import torch.nn as nn


def get_optimizer(cfg, params):
    params_dict = cfg.get("params", {})
    try:
        optim_class = getattr(torch.optim, cfg['algo'])
    except AttributeError:
        raise ValueError(f"Alorithm '{cfg['algo']}' does not exist in torch.optim")
    return optim_class(params, **params_dict)