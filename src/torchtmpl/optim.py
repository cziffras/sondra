import inspect

import torch


def get_optimizer(cfg, params):
    raw_params = cfg["optimizer"].get("params", {})

    # a list means a tuple argument such as AdamW's betas; dropping it, as a
    # plain float() filter did, silently ignored what the config asked for
    params_dict = {
        k: tuple(float(x) for x in v) if isinstance(v, list) else float(v)
        for k, v in raw_params.items()
    }

    try:
        optim_class = getattr(torch.optim, cfg["optimizer"]["name"])
    except AttributeError:
        raise ValueError(f"Algorithm '{cfg['optimizer']['name']}' does not exist in torch.optim")
    return optim_class(params, **params_dict)


def get_scheduler(cfg, optimizer, steps_per_epoch=None):
    """
    Build a scheduler from the config, forwarding only the arguments its
    signature accepts.
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
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    except AttributeError as e:
        raise ValueError(
            f"Scheduler '{scheduler_name}' does not exist in torch.optim.lr_scheduler"
        ) from e

    sig = inspect.signature(scheduler_class)
    valid_args = {k: v for k, v in all_args.items() if k in sig.parameters}

    return scheduler_class(**valid_args)
