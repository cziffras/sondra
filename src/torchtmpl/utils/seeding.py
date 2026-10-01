import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Seed every generator the run draws from, this is needed to make the batch
    order and augmentations deterministic. The data splitting is deterministic
    by design (see mosaic splitting). We feared that it was necessary to seed
    all processes spawned by torch.utils.data.DataLoader, which in fact is not:
    PyTorch gives each one `base_seed + worker_id` for `random`, `torch` and 
    `numpy`.

    Seeding alone does not make two runs identical: by default cuDNN may pick
    kernels whose threading causes roundoff variations, see :

    https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
