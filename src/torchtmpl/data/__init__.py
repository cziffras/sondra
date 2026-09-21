from .wrappers import Dataloaders, get_polsf_dataloaders

__all__ = ["Dataloaders", "get_dataloaders"]

_DATASETS = {
    "polsf": get_polsf_dataloaders,
}


def get_dataloaders(config, use_cuda):
    """Build the dataloaders for the dataset named in the config."""
    name = config["data"]["dataset"]
    try:
        builder = _DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}") from None

    return builder(config, use_cuda)
