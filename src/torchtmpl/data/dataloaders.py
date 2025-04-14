import torch
import logging
from torchcvnn.transforms import LogAmplitude, PolSARtoTensor
import torchvision.transforms.v2 as v2

from .PolSF import WrappedPolSFDataset, PolSFContrastive
from ..utils import get_transform, _pri
from ..transforms import SARContrastiveAugmentations


def get_polsf_dataloaders(data_config, use_cuda, contrastive=False, debug=False):

    config = {
        "valid_ratio": data_config["valid_ratio"],
        "batch_size": data_config["batch_size"],
        "num_workers": data_config["num_workers"],
        "root_dir": data_config["root_dir"],
        "patch_size": tuple(data_config.get("patch_size", (128, 128))),
    }
    config["patch_stride"] = tuple(data_config.get("patch_stride", config["patch_size"]))
    
    # Création du dataset selon le mode
    if contrastive:
        # Simplification de la création du transform contrastif
        transform_args = data_config.get("transform_contrastive", {})
        transform_class = SARContrastiveAugmentations(**transform_args.get("params", {}))
        
        dataset = PolSFContrastive(
            rootdir=config["root_dir"],
            patch_size=config["patch_size"],
            patch_stride=config["patch_stride"],
            transform_contrastive=v2.Compose([PolSARtoTensor(), transform_class, LogAmplitude()]),
        )
    else:
        dataset = WrappedPolSFDataset(
            root=config["root_dir"],
            patch_size=config["patch_size"],
            patch_stride=config["patch_stride"],
            transform=v2.Compose([PolSARtoTensor()]),
        )

    # Affichage d'informations de debug si nécessaire
    if debug:
        _print_dataset_debug_info(dataset, config)

    logging.info(f"Loaded {len(dataset)} samples")

    # Division en train/val et création des dataloaders
    train_dataset, valid_dataset = _split_dataset(dataset, config["valid_ratio"])
    
    # Application des transformations supplémentaires si nécessaire
    if not contrastive:
        transform = get_transform(data_config)
        train_dataset.dataset.set_augment_transform(v2.Compose([transform, LogAmplitude()]))
        valid_dataset.dataset.set_augment_transform(LogAmplitude())

    # Création des dataloaders
    loader_params = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": use_cuda,
    }
    
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_params)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **loader_params)

    # Détermination des informations de sortie
    num_classes = 1 if contrastive else len(dataset.classes)
    input_size = tuple(dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes


def get_full_image_dataloader(data_config):
    """Crée un dataloader pour traiter les images complètes"""
    # Création du dataset et récupération des dimensions
    base_dataset = WrappedPolSFDataset(
        root=data_config["root_dir"],
        patch_size=tuple(data_config.get("patch_size", (128, 128))),
        patch_stride=tuple(data_config.get("patch_stride", data_config.get("patch_size", (128, 128)))),
        transform=v2.Compose([PolSARtoTensor(), LogAmplitude()]),
    )
    
    # Création du dataloader
    loader = torch.utils.data.DataLoader(
        GenericDatasetWrapper(base_dataset),
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )
    
    return loader, base_dataset.alos_dataset.nsamples_per_cols, base_dataset.alos_dataset.nsamples_per_rows