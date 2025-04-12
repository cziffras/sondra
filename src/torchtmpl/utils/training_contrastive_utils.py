# Standard imports
import os
from typing import Tuple
import inspect
import warnings

# External imports
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable
import numpy as np


def train_one_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    epoch: int,
    max_norm=2.5,
    loss_weights=None,
) -> dict:
    """
    Run the training loop for nsteps minibatches of the dataloader

    Arguments:
        model: the model to train
        loader: an iterable dataloader
        f_loss (nn.Module): the loss
        optim : an optimizing algorithm
        device: the device on which to run the code

    Returns:
        A dictionary with averaged training metrics
    """
    model.train()

    loss_avg = 0
    gradient_norm = 0
    num_samples = 0
    num_batches = 0

    for data in tqdm.tqdm(loader):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs_1, inputs_2 = data
            inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        else:
            raise ValueError("Contrastive Dataloader not returning a tuple of samples")

        # Forward propagate through the model
        out_1, out_2 = model(inputs_1), model(inputs_2)

        loss = 0.0 
        if isinstance(out_1, list) and isinstance(out_2, list) and len(out_1) == len(out_2):
            if loss_weights is not None:
                loss_weights = loss_weights.tolist()
                for i in range(len(out_1)):
                    loss += loss_weights[i] * f_loss(out_1[i], out_2[i])
                loss /= sum([w.item() for w in loss_weights])

            else:
                for i in range(len(out_1)):
                    loss += f_loss(out_1[i], out_2[i])
                loss /= len(out_1) 
        else:
            loss += f_loss(out_1, out_2)

        # Backward pass and update
        optim.zero_grad()
        loss.backward()

        # Clip gradients to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm, norm_type=2
        )

        # Compute the norm of the gradients
        total_norm = np.sqrt(
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        )
        gradient_norm += total_norm

        optim.step()
        if isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.CyclicLR,
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.CosineAnnealingLR,
            ),
        ):
            scheduler.step()
        elif isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            scheduler.step(epoch + num_batches / len(loader))

        num_samples += inputs_1.shape[0]
        num_batches += 1
        loss_avg += inputs_1.shape[0] * loss.item()

        del loss, inputs_1, inputs_2

    torch.cuda.empty_cache()

    metrics = {
        "train_loss": loss_avg / num_samples,
        "gradient_norm": gradient_norm / num_batches,
    }

    return metrics


def valid_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run the training loop for nsteps minibatches of the dataloader

    Arguments:
        model: the model to train
        loader: an iterable dataloader
        f_loss (nn.Module): the loss
        optim : an optimizing algorithm
        device: the device on which to run the code

    Returns:
        A dictionary with averaged training metrics
    """
    model.eval()

    loss_avg = 0
    num_samples = 0
    num_batches = 0

    for data in tqdm.tqdm(loader):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs_1, inputs_2 = data
            inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        else:
            raise ValueError("Contrastive Dataloader not returning a tuple of samples")

        # Forward propagate through the model
        out_1, out_2 = model(inputs_1), model(inputs_2)

        loss = 0.0 
        if isinstance(out_1, list) and isinstance(out_2, list) and len(out_1) == len(out_2):
            if loss_weights is not None:
                loss_weights = loss_weights.tolist()
                for i in range(len(out_1)):
                    loss += loss_weights[i] * f_loss(out_1[i], out_2[i])
                loss /= sum([w.item() for w in loss_weights])

            else:
                for i in range(len(out_1)):
                    loss += f_loss(out_1[i], out_2[i])
                loss /= len(out_1) 
        else:
            loss += f_loss(out_1, out_2)

        num_samples += inputs_1.shape[0]
        num_batches += 1
        loss_avg += inputs_1.shape[0] * loss.item()

        del loss, inputs_1, inputs_2

    torch.cuda.empty_cache()

    metrics = {
        "valid_loss": loss_avg / num_samples,
    }

    return metrics
