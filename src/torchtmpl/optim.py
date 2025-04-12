import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
)


def get_optimizer(cfg, params):
    params_dict = cfg["optimizer"].get("params", {})
    try:
        optim_class = getattr(torch.optim, cfg["optimizer"]["name"])
    except AttributeError:
        raise ValueError(f"Algorithm '{cfg['optimizer']['name']}' does not exist in torch.optim")
    return optim_class(params, **params_dict)


def get_scheduler(cfg, optimizer, steps_per_epoch):
    scheduler_name = cfg["scheduler"]["name"]
    if scheduler_name == "OneCycleLR":
        cfg["scheduler"]["params_onecyclelr"]["steps_per_epoch"] = steps_per_epoch
        cfg["scheduler"]["params_onecyclelr"]["epochs"] = cfg["nepochs"]
        return OneCycleLR(optimizer, **cfg["scheduler"]["params_onecyclelr"])
    elif scheduler_name == "CosineAnnealingLR":
        cfg["scheduler"]["params_cosinelr"]["T_max"] = cfg["nepochs"]
        return CosineAnnealingLR(optimizer, **cfg["scheduler"]["params_cosinelr"])
    elif scheduler_name is None:
        return None 
    else:
        return eval(
            f"torch.optim.lr_scheduler.{scheduler_name}(optimizer, **cfg['scheduler']['params'])"
        )