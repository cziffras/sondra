import torch
import numpy as np
from typing import Union

class ToTensor:
    def __init__(self, dtype: torch.dtype = torch.complex64):
        self.dtype = dtype

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Convert numpy array to PyTorch tensor and Rearrange dimensions from HWC to CHW .permute(2, 0, 1)
        tensor = torch.from_numpy(image).to(self.dtype)
        return tensor

class PolSARtoTensor:
    """
    Transform a PolSAR image into a 3D torch tensor.
    """
    def __call__(self, element: Union[np.ndarray, dict]) -> torch.Tensor:

        if isinstance(element, np.ndarray):
            assert len(element.shape) == 3, "Element should be a 3D numpy array"
            if element.shape[0] == 3:
                return self._create_tensor(element[0], element[1], element[2])
            elif element.shape[0] == 2:
                return self._create_tensor(element[0], element[1])
            elif element.shape[0] == 4:
                return self._create_tensor(element[0], element[1], element[2], element[3])
            
        elif isinstance(element, dict):
            if len(element) == 3:
                return self._create_tensor(element["HH"], element["HV"], element["VV"])
            elif len(element) == 2:
                if "HH" in element:
                    return self._create_tensor(element["HH"], element["HV"])
                elif "VV" in element:
                    return self._create_tensor(element["HV"], element["VV"])
                else:
                    raise ValueError(
                        "Dictionary should contain keys HH, HV, VV or HH, VV"
                    )
            elif len(element) == 4:
                return self._create_tensor(
                    element["HH"], element["HV"], element["VH"], element["VV"]
                )
            else:
                raise ValueError("Element should be a numpy array or a dictionary")

    def _create_tensor(self, *channels) -> torch.Tensor:
        return torch.as_tensor(
            np.stack(channels, axis=-1).transpose(2, 0, 1),
            dtype=torch.complex64,
        )