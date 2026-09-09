import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentKendall(nn.Module):
    """
    Learnable linear combination (Kendall & Gal, 2018) of multi stage NT‑Xent
    
    Args: 
        `num_stages` : nb of stages (4 in this work)
        `lambda_l2` : weighing of the L² penalty on log sigma_i²
    """
    def __init__(self,
                 num_stages: int,
                 temperature: float = 0.1,
                 eps: float = 1e-6,
                 lambda_reg: float = 1.0):
        super().__init__()
        self.nt_xent   = NTXentLoss(temperature, eps)
        self.log_vars  = nn.Parameter(torch.zeros(num_stages), requires_grad=True)  # log \sigma_i²
        self.lambda_l2 = lambda_reg

    def forward(self,
                zs1: list[torch.Tensor],   # list of tensors (B,D)
                zs2: list[torch.Tensor]) -> torch.Tensor:

        if len(zs1) != len(zs2):
            raise ValueError("Embeddings lists must have same length")

        losses = torch.stack([self.nt_xent(z1, z2) for z1, z2 in zip(zs1, zs2)], dim=0)
        precision = torch.exp(-self.log_vars)            # 1/ \sigma_i²

        loss = (precision * losses).sum()                
        loss += self.log_vars.sum()                  
        loss += self.lambda_l2 * (self.log_vars ** 2).sum()  # L² penalty

        return loss

    @property
    def weights(self) -> torch.Tensor:
        "Returns normalised weights from log_vars to monitor collapsing"
        precision = torch.exp(-self.log_vars)
        return (precision / precision.sum()).detach()

class NTXentKLUnif(nn.Module):
    """
    Learnable linear combination of multi stage NT‑Xent with KL div regularization
    
    Args: 
        `num_stages` : nb of stages (4 in this work)
        `lambda_kl` : weighing of the KL div against uniform distribution penalty 
    """
    def __init__(self,
                 num_stages: int,
                 temperature: float = 0.1,
                 eps: float = 1e-6,
                 lambda_reg: float = 1):
        super().__init__()
        self.nt_xent   = NTXentLoss(temperature, eps)
        self.logits    = nn.Parameter(torch.zeros(num_stages), requires_grad=True)  # logits w_i
        self.lambda_kl = lambda_reg
        self.N = num_stages

    def forward(self,
                zs1: list[torch.Tensor],
                zs2: list[torch.Tensor]) -> torch.Tensor:

        if len(zs1) != len(zs2):
            raise ValueError("Embeddings lists must have same length")
    
        w  = F.softmax(self.logits.real, dim=0)         # (N,) – real, >=0, somme=1
        losses = torch.stack([self.nt_xent(z1, z2) for z1, z2 in zip(zs1, zs2)])  # (N,)

        # fusion + KL regularization 
        kl = torch.sum(w * torch.log(w * self.N + 1e-12)) # KL(w || U)

        return torch.sum(w * losses) + self.lambda_kl * kl

    @property
    def weights(self) -> torch.Tensor:
        "Returns normalised weights to monitor collapsing w_i"
        return F.softmax(self.logits, dim=0).detach()

class NTXentLearnableTemp(nn.Module):
    """
    See notes.md, section "Third try : the learnable temperature"
    """
    def __init__(self,
                 num_stages: int,
                 init_temperature: float = 0.1,
                 eps: float = 1e-6,
                 lambda_reg: float = 1.0):
        super().__init__()
        self.nt_xent  = NTXentLoss(eps=eps)
        init_log_beta = -torch.log(torch.tensor(float(init_temperature)))
        self.log_beta = nn.Parameter(torch.full((num_stages,), init_log_beta.item()))
        self.lambda_kl = lambda_reg
        self.N = num_stages

    def forward(self,
                zs1: list[torch.Tensor],
                zs2: list[torch.Tensor]) -> torch.Tensor:

        if len(zs1) != len(zs2):
            raise ValueError("Embeddings lists must have same length")

        beta = torch.exp(self.log_beta)
        losses = torch.stack(
            [self.nt_xent(z1, z2, temperature=1.0 / b) for z1, z2, b in zip(zs1, zs2, beta)],
            dim=0,
        )

        w  = beta / beta.sum()
        kl = torch.sum(w * torch.log(w * self.N + 1e-12))

        return losses.sum() + self.lambda_kl * kl

    @property
    def weights(self) -> torch.Tensor:
        beta = torch.exp(self.log_beta)
        return (beta / beta.sum()).detach()

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss with regularization features
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                temperature: float | torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z1, z2: Tensors of shape (N, D) loss supports type complex
        """
        # Concat
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # L2 norm with epsilon
        # ||z_i|| = sqrt(sum |z_i|^2) clamp with eps min
        norm = torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(self.eps)
        z_normalized = z / norm  # shape (2N, D)

        # similarity with module of hermitian products
        sim = torch.abs(torch.matmul(z_normalized, z_normalized.conj().T))  # (2N, 2N)
        # clamp to avoid overflow
        sim = sim.clamp(-1 + self.eps, 1 - self.eps)

        N = z1.size(0)
        device = z.device
        diag = torch.arange(N, device=device)

        # extract positive pairs
        pos_1 = sim[diag, diag + N]       # (N,)
        pos_2 = sim[diag + N, diag]       # (N,)
        positives = torch.cat([pos_1, pos_2], dim=0).unsqueeze(1)  # (2N,1)

        # mask to exclude diagonals
        # True where i != j, False on diagonal
        mask = ~torch.eye(2 * N, device=device, dtype=torch.bool)

        tau = self.temperature if temperature is None else temperature

        # Logits = sim / temperature, we mask diag then logsumexp
        logits = sim / tau
        # on met très bas les diagonnales pour qu'elles n'entrent pas dans logsumexp
        logits_masked = logits.masked_fill(~mask, float("-inf"))

        # Loss for each row
        # -log(exp(sim_pos/T) / sum(exp(sim_all/T)))
        # = - (sim_pos/T) + logsumexp(sim_all/T)
        loss_per_sample = -positives / tau + torch.logsumexp(logits_masked, dim=1, keepdim=True)

        # mean on all pairs
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

def get_loss(lossname, **kwargs):
    if lossname == "NTXentLoss":
        return NTXentLoss(**kwargs)
    if lossname == "NTXentKendall":
        return NTXentKendall(**kwargs)
    if lossname == "NTXentKLUnif":
        return NTXentKLUnif(**kwargs)
    if lossname == "NTXentLearnableTemp":
        return NTXentLearnableTemp(**kwargs)
    if lossname == "FocalLoss":
        return FocalLoss(**kwargs)
    try:
        return getattr(nn, lossname)(**kwargs)
    except AttributeError:
        raise ValueError(f"Loss '{lossname}' unknown in torch.nn.")
