import inspect
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


# def get_scheduler(cfg, optimizer, steps_per_epoch):
#     scheduler_name = cfg["scheduler"]["name"]
#     if scheduler_name == "OneCycleLR":
#         cfg["scheduler"]["params_onecyclelr"]["steps_per_epoch"] = steps_per_epoch
#         cfg["scheduler"]["params_onecyclelr"]["epochs"] = cfg["nepochs"]
#         return OneCycleLR(optimizer, **cfg["scheduler"]["params_onecyclelr"])
#     elif scheduler_name == "CosineAnnealingLR":
#         cfg["scheduler"]["params_cosinelr"]["T_max"] = cfg["nepochs"]
#         return CosineAnnealingLR(optimizer, **cfg["scheduler"]["params_cosinelr"])
#     elif scheduler_name is None:
#         return None 
#     else:
#         return eval(
#             f"torch.optim.lr_scheduler.{scheduler_name}(optimizer, **cfg['scheduler']['params'])"
#         )

def get_scheduler(cfg, optimizer, steps_per_epoch=None):
    """
    Récupère un scheduler basé sur la configuration en passant
    automatiquement les paramètres appropriés via introspection.
    """
    scheduler_name = cfg["scheduler"]["name"]
    
    if scheduler_name is None:
        return None
    
    all_args = {
        "optimizer": optimizer,
    }
    
    if "params" in cfg["scheduler"]:
        params = cfg["scheduler"]["params"].copy()
        
        if scheduler_name == "OneCycleLR" and steps_per_epoch is not None:
            params["steps_per_epoch"] = steps_per_epoch
            params["epochs"] = cfg["nepochs"]
        elif scheduler_name == "CosineAnnealingLR":
            params["T_max"] = cfg["nepochs"]
            
        all_args.update(params)
    
    try:
        if scheduler_name in ["OneCycleLR", "CosineAnnealingLR"]:
            scheduler_class = eval(scheduler_name)
        else:
            scheduler_class = eval(f"torch.optim.lr_scheduler.{scheduler_name}")
        
        sig = inspect.signature(scheduler_class)
        valid_args = {k: v for k, v in all_args.items() if k in sig.parameters}
        
        return scheduler_class(**valid_args)
    
    except (NameError, AttributeError) as e:
        raise ValueError(f"Scheduler '{scheduler_name}' non reconnu ou non importé: {e}")