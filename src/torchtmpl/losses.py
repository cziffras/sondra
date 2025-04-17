import torch 
import torch.nn as nn 
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss améliorée pour embeddings réels ou complexes.
    Stabilisation via epsilons et clamp.
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: Tensors de shape (N, D), peuvent être complexes.
        Returns:
            perte scalaire moyenne.
        """
        # 1) Concaténation
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # 2) Normalisation L2 (avec epsilon)
        #    ||z_i|| = sqrt(sum |z_i|^2) clamp à eps min
        norm = torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(self.eps)
        z_normalized = z / norm  # shape (2N, D)

        # 3) Matrice de similarité cosinus
        #    si complexe, on prend la partie réelle
        sim = torch.matmul(z_normalized, z_normalized.conj().T).real  # (2N, 2N)
        # on clamp les valeurs extrêmes pour éviter overflow
        sim = sim.clamp(-1 + self.eps, 1 - self.eps)

        N = z1.size(0)
        device = z.device
        diag = torch.arange(N, device=device)

        # extract positive pairs
        pos_1 = sim[diag, diag + N]       # (N,)
        pos_2 = sim[diag + N, diag]       # (N,)
        positives = torch.cat([pos_1, pos_2], dim=0).unsqueeze(1)  # (2N,1)

        # mask to exclude diagonals
        #    True where i != j, False on diagonale
        #    on veut un mask booléen pour logsumexp
        mask = ~torch.eye(2 * N, device=device, dtype=torch.bool)

        # 6) Logits = sim / temperature, on masque diag puis logsumexp
        logits = sim / self.temperature
        # on met très bas les diagonnales pour qu'elles n'entrent pas dans logsumexp
        logits_masked = logits.masked_fill(~mask, float("-inf"))

        # 7) Calcul de la perte pour chaque ligne
        #    -log(exp(sim_pos/τ) / sum(exp(sim_all/τ)))
        #    = - (sim_pos/τ) + logsumexp(sim_all/τ)
        loss_per_sample = -positives / self.temperature + torch.logsumexp(logits_masked, dim=1, keepdim=True)

        # 8) Moyenne sur toutes les paires
        return loss_per_sample.mean()

class FocalLoss(nn.Module):
    def __init__(
            self, 
            alpha=torch.tensor([2.38, 40.0, 3.03, 27.77, 10.27, 15.24, 55.0]), # torch.tensor([1.0, 1.46, 0.11, 1.19, 0.40, 0.54, 2.29] 
            ignore_index=0, 
            gamma=1.2
        ):

        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        # Convert alpha to a tensor if it's provided as a list/array.
        if alpha is not None:
            self.alpha = alpha.clone().detach()
        else:
            self.alpha = None

    def forward(self, softmax_probs, targets):
        """
        Args:
            softmax_probs: Tensor of precomputed softmax probabilities.
            targets: Ground truth labels.
        """
        # Prevent log(0) issues
        softmax_probs = torch.clamp(softmax_probs, min=1e-10, max=1.0)
        log_probs = torch.log(softmax_probs)

        # Ensure targets are int64 (required for indexing and loss functions)
        targets = targets.type(torch.int64)

        # Compute cross-entropy loss (per example) with ignore_index handling
        ce_loss = F.nll_loss(
            log_probs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Recover the probability for the true class
        pt = torch.exp(-ce_loss)

        # Compute the focal modulation term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # If alpha is provided, apply per-class weighting
        if self.alpha is not None:
            # Ensure alpha is on the same device as softmax_probs
            alpha = self.alpha.to(softmax_probs.device)
            # Index into alpha using the target labels
            alpha_weights = alpha[targets]
            loss = alpha_weights * loss

        # Create a mask to only average over non-ignored indices
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss[valid_mask]

        return loss.mean()

def get_loss(lossname, **loss_kwargs):
    "Use CrossEntropyLoss as a baseline for segmentation task"
    if lossname=="NTXentLoss":
        return NTXentLoss(**loss_kwargs)
    elif lossname=="FocalLoss":
        return FocalLoss(**loss_kwargs)
    try:
        loss_class = getattr(nn, lossname)
    except AttributeError:
        raise ValueError(f"Loss '{lossname}' does not exist in torch.nn.")
    return loss_class(**loss_kwargs)