# Standard imports
import os
from typing import Tuple
import inspect
import warnings

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
import numpy as np


def train_one_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_norm: float = 2.5,
    lambda_kl: float = 0.1,
    epoch: int = 0,
) -> dict:
    model.train()

    loss_sum = 0.0
    gradient_norm_sum = 0.0
    num_samples = 0
    num_batches = 0

    for x1, x2 in tqdm.tqdm(loader, desc=f"Train Epoch {epoch}"):
        x1, x2 = x1.to(device), x2.to(device)
        zs1 = model(x1)  # list of (b, proj_dim)
        zs2 = model(x2)

        # NT-Xent for each stage
        losses = torch.stack([f_loss(z1, z2) for z1, z2 in zip(zs1, zs2)], dim=0)

        # Learnable fusion of losses (again, instead of defining weights in config)
        precision = torch.exp(-model.log_vars)   # 1/σ_i²
        contrastive_loss = torch.sum(precision * losses)
        
        # régularisation log σ_i (cf. Kendall & Gal, CVPR 2018)
        loss = contrastive_loss + torch.sum(model.log_vars)

        # backward + clipping
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)

        # gradient norm L2
        total_norm = np.sqrt(
            sum(p.grad.detach().norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None)
        )
        gradient_norm_sum += total_norm

        # update
        optim.step()

        with torch.no_grad():
            model.log_vars.data.clamp_(-5.0, 5.0) 

        # step scheduler
        if isinstance(scheduler, (torch.optim.lr_scheduler.CyclicLR,
                                  torch.optim.lr_scheduler.OneCycleLR,
                                )):
            scheduler.step()

        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        
            scheduler.step(epoch + num_batches / len(loader))

        # stats
        batch_size = x1.size(0)
        loss_sum += loss.item() * batch_size
        num_samples += batch_size
        num_batches += 1

    torch.cuda.empty_cache()

    metrics = {
        "train_loss": loss_sum / num_samples,
        "avg_grad_norm": gradient_norm_sum / num_batches,
    }

    return metrics


def valid_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run one validation epoch for contrastive learning.

    Arguments:
        model: the SegFormer model with contrastive=True
        loader: DataLoader returning pairs (x1, x2)
        f_loss: NT-Xent loss module
        device: torch device

    Returns:
        A dict with averaged validation loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for x1, x2 in tqdm.tqdm(loader, desc="Valid"):
            x1, x2 = x1.to(device), x2.to(device)

            zs1 = model(x1)  # list of embeddings [(b, D), …]
            zs2 = model(x2)

            # Per-stage NT-Xent
            losses = torch.stack([
                f_loss(z1, z2)
                for z1, z2 in zip(zs1, zs2)
            ], dim=0)  # shape = (num_stages,)

            # Pondération par incertitude homoscédastique (Kendall & Gal, 2018)
            precision = torch.exp(-model.log_vars)          # 1/σ_i²
            loss = (precision * losses).sum() + model.log_vars.sum()

            batch_size = x1.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    return {"valid_loss": total_loss / num_samples}