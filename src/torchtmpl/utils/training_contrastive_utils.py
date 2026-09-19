import numpy as np
import torch
import tqdm
from torch import nn


def train_one_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_norm: float = 2.5,
    lambda_reg: float = 1,
    epoch: int = 0,
) -> dict:
    model.train()

    loss_sum = 0.0
    gradient_norm_sum = 0.0
    num_samples = 0
    num_batches = 0

    loss_params = [p for p in f_loss.parameters() if p.requires_grad]
    existing = {p for g in optim.param_groups for p in g["params"]}

    to_add = [p for p in loss_params if p not in existing]
    if to_add:
        optim.add_param_group({"params": to_add})

    for x1, x2 in tqdm.tqdm(loader, desc=f"Train Epoch {epoch}"):
        x1, x2 = x1.to(device), x2.to(device)

        zs1 = model(x1)
        zs2 = model(x2)

        if hasattr(f_loss, "lambda_reg"):
            f_loss.lamda_reg = float(lambda_reg)

        loss = f_loss(zs1, zs2)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)

        total_norm = np.sqrt(
            sum(
                p.grad.detach().norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        )
        gradient_norm_sum += total_norm

        optim.step()
        if hasattr(f_loss, "log_vars"):
            with torch.no_grad():
                f_loss.log_vars.data.clamp_(-5.0, 5.0)
        if hasattr(f_loss, "log_beta"):
            with torch.no_grad():
                f_loss.log_beta.data.clamp_(-5.0, 5.0)

        if isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.CyclicLR,
                torch.optim.lr_scheduler.OneCycleLR,
            ),
        ):
            scheduler.step()

        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + num_batches / len(loader))

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


@torch.no_grad()
def valid_contrastive_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    device: torch.device,
    lambda_reg: float = 1,
) -> dict:

    model.eval()
    f_loss.eval()

    if hasattr(f_loss, "lambda_reg"):
        f_loss.lamda_reg = float(lambda_reg)

    total_loss = num_samples = 0

    for x1, x2 in tqdm.tqdm(loader, desc="Valid"):
        x1, x2 = x1.to(device), x2.to(device)
        zs1, zs2 = model(x1), model(x2)
        loss = f_loss(zs1, zs2)

        bsz = x1.size(0)
        total_loss += loss.item() * bsz
        num_samples += bsz

    return {"valid_loss": total_loss / num_samples}
