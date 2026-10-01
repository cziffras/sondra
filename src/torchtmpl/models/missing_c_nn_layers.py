import torch
import torch.fx
from torch import Tensor, nn
from torchcvnn.nn import modules as c_nn
from torchvision.utils import _log_api_usage_once


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(stochastic_depth)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim

    if torch.is_complex(input):
        if input.dtype == torch.complex64:
            float_dtype = torch.float32
        elif input.dtype == torch.complex128:
            float_dtype = torch.float64
        else:
            raise ValueError("Unsupported complex dtype for input")
    else:
        float_dtype = input.dtype

    noise = torch.empty(size, dtype=float_dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class BatchNorm2d(c_nn.BatchNorm2d):
    """
    torchcvnn's complex BatchNorm2d, did not take out the autograd the running
    statistics, causing an OOM on my personal computer this translated as a growing
    RAM usage over epochs while not the GPU's memory meaning that Pytorch was somehow
    storing history objects in the RAM (backward frees the VRAM from large tensors).
    """

    def forward(self, z: Tensor) -> Tensor:
        out = super().forward(z)
        if self.training and self.track_running_stats:
            self.running_mean = self.running_mean.detach()
            self.running_var = self.running_var.detach()
        return out



class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2d tensors with complex parameters.

    WARNING :

    1 -
        torchcvnn implementation of LayerNorm seems incorrect, it does : 

        ```python
        z_ravel = z.view(-1, C).transpose(0, 1)   # (C, B·H·W)
        mus = z_ravel.mean(axis=-1)               # a mean PER CHANNEL, over the B·H·W rows
        ```

        This has a major consequence : in training the model sees random patches while in test
        it sees them in the spatial order, the statistics differs a lot, so the features are 
        distorted and so the predictions ! Need to check it further. 

        2 -
            Layernorm 2D might not be well suited for SAR data. A LayerNorm brings the
            channels of each pixel back to the same order of magnitude and thus obliterate
            "brightness" information : after the first convolution, which has no bias, a water
            pixel and an urban pixel differ mostly by that magnitude. The per-pixel version indeed learns badly. 



    NOTE : This class is kept for this documentation only, nothing uses it any more.
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = c_nn.LayerNorm(normalized_shape)

    def forward(self, x: Tensor) -> Tensor:
        shape_orig = x.shape
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C) layernorm applies averaging onto channels/the last axis
        x = x.reshape(-1, x.size(-1)) # -> (B·H·W, C) : one row per pixel of the whole batch
        x = self.ln(x) # READ DOCSTRING
        x = x.view(shape_orig[0], shape_orig[2], shape_orig[3], shape_orig[1])
        x = x.permute(0, 3, 1, 2)
        return x