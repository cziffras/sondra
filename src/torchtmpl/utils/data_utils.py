import numpy as np
import torch


class ToTensor:
    def __init__(self, dtype: torch.dtype = torch.complex64):
        self.dtype = dtype

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).to(self.dtype)
        return tensor
