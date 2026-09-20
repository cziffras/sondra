from typing import NamedTuple

import torch

from .PolSF import PolSFDataManager


class Dataloaders(NamedTuple):
    train: torch.utils.data.DataLoader
    valid: torch.utils.data.DataLoader
    input_size: tuple[int, ...]
    num_classes: int
    classes: list[str]


def get_polsf_dataloaders(data_config, use_cuda, contrastive=False):
    manager = PolSFDataManager(config=data_config, use_cuda=use_cuda)
    return Dataloaders(*manager.get_dataloaders(contrastive=contrastive))


def get_full_image_dataloader(data_config, use_cuda, contrastive=False):
    manager = PolSFDataManager(config=data_config, use_cuda=use_cuda)
    return manager.get_full_image_dataloader()
