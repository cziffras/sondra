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

