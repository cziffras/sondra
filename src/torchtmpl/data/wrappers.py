from typing import NamedTuple

import torch

from .PolSF import PolSFDataManager


class Dataloaders(NamedTuple):
    """
    `valid` and `test` are None in contrastive mode, which pre-trains on every
    patch and holds nothing out. `classes` is empty there for the same reason.
    """

    train: torch.utils.data.DataLoader
    valid: torch.utils.data.DataLoader | None
    test: torch.utils.data.DataLoader | None
    input_size: tuple[int, ...]
    num_classes: int
    classes: list[str]


def get_polsf_dataloaders(config, use_cuda):
    manager = PolSFDataManager(config, use_cuda=use_cuda)
    return Dataloaders(*manager.get_dataloaders())


def get_full_image_dataloader(config, use_cuda):
    manager = PolSFDataManager(config, use_cuda=use_cuda)
    return manager.get_full_image_dataloader()
