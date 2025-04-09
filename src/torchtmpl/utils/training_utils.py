# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm
import numpy as np


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train_one_epoch(model, loader, f_loss, optimizer, device, accumulation_steps=1):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- A torch.device

    Returns :
        The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward
        loss = loss / accumulation_steps
        loss.backward()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Train loss : {total_loss/num_samples:.4f}")

    return total_loss / num_samples


def train_one_contrastive_epoch(
    model, loader, f_loss, optimizer, device, accumulation_steps=1
):
    """
    Train a model for one epoch using contrastive learning,
    iterating over the loader using the f_loss to compute the loss
    and the optimizer to update the parameters of the model.
    Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a contrastive loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- A torch.device
        accumulation_steps -- Number of steps for gradient accumulation

    Returns :
        The averaged train metrics computed over the loader
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    pbar = tqdm.tqdm(enumerate(loader))

    for i, (x1, x2) in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)

        loss = f_loss(z1, z2)

        # Backward
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * x1.shape[0] 
        num_samples += x1.shape[0]

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Train loss : {total_loss/num_samples:.4f}")

    return total_loss / num_samples 


def compute_accuracy_from_polsar_dataset(model, dataset, device):
    """
    Compute the accuracy of a model on a PolSAR dataset
    Arguments:
        model     -- A torch.nn.Module object (classification model)
        dataset   -- A torch.utils.data.Dataset with PolSAR data
        device    -- A torch.device

    Returns:
        The accuracy of the model on the dataset
    """
    
    
    raise NotImplementedError

