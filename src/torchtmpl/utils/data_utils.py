import torch
import numpy as np
from typing import Union
import random

class ToTensor:
    def __init__(self, dtype: torch.dtype = torch.complex64):
        self.dtype = dtype

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Convert numpy array to PyTorch tensor and Rearrange dimensions from HWC to CHW .permute(2, 0, 1)
        tensor = torch.from_numpy(image).to(self.dtype)
        return tensor


def _get_transform(data_config):
    transform_args = data_config.get("transform", {})
    transform_name = transform_args.get("name", "SARContrastiveAugmentations")
    transform_params = transform_args.get("params", {})

    return eval(f"{transform_name}")(**transform_params)

def _split_dataset(dataset, valid_ratio):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(valid_ratio * len(dataset))
    
    return (
        torch.utils.data.Subset(dataset, indices[split:]),
        torch.utils.data.Subset(dataset, indices[:split]),
    )

def _print_dataset_debug_info(dataset, config):
    """Affiche des informations de debug sur le dataset"""
    print("\n========== Dataset Debug Info ==========")
    try:
        first_item = dataset[0]
        patch = first_item[0] if isinstance(first_item, tuple) else first_item
            
        print(f"Patch shape: {patch.shape}")
        print(f"Patch size: {config['patch_size']}")
        print(f"Patch stride: {config['patch_stride']}")
        print(f"Total patches: {len(dataset)}")
    except Exception as e:
        print(f"Error accessing patch: {e}")
    print("========================================\n")