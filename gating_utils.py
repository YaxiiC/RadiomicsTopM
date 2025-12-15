import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps: float = 1e-20, device=None):
    """Sample Gumbel(0, 1) noise."""
    if device is None:
        device = torch.device("cpu")
    u = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)


def hard_topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return a k-hot mask with exactly k ones per row."""
    topk = torch.topk(logits, k, dim=1).indices
    mask = torch.zeros_like(logits)
    mask.scatter_(1, topk, 1.0)
    return mask


def ste_topk_mask(logits: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Straight-through estimator for hard top-k selection."""
    z_hard = hard_topk_mask(logits, k)
    s = torch.sigmoid(logits / tau)
    return z_hard + (s - s.detach())


def gumbel_topk_ste_mask(logits: torch.Tensor, k: int, tau: float, lam: float) -> torch.Tensor:
    """Gumbel-TopK straight-through estimator.

    Args:
        logits: [..., N]
        k: number of elements to keep
        tau: temperature for the sigmoid relaxation
        lam: noise strength for Gumbel exploration
    """

    if lam > 0:
        g = sample_gumbel(logits.shape, device=logits.device)
        logits_noisy = logits + lam * g
    else:
        logits_noisy = logits

    z_hard = hard_topk_mask(logits_noisy, k)
    s = torch.sigmoid(logits / tau)
    return z_hard + (s - s.detach())
