from .wrappers import Dataloaders, get_polsf_dataloaders

__all__ = ["Dataloaders", "get_dataloaders"]

_DATASETS = {
    "polsf": get_polsf_dataloaders,
}


def get_dataloaders(data_config, use_cuda, contrastive=False):
    """Build the train/valid dataloaders for the dataset named in the config."""
    name = data_config["dataset"]
    try:
        builder = _DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}") from None

    return builder(data_config, use_cuda, contrastive=contrastive)
