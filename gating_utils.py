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


def compose_continuous_topm_weights(z_hard: torch.Tensor, w_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose continuous Top-M weights with a single soft factor in the forward pass.

    Args:
        z_hard: Hard top-M mask (k-hot).
        w_soft: Soft weights computed from logits / temperature.

    Returns:
        weights: Forward weights using ``z_hard * w_soft`` while attaching gradients via STE.
        z_st: Straight-through mask used for gradient flow.
    """

    z_st = z_hard + (w_soft - w_soft.detach())
    weights_fwd = z_hard * w_soft

    # Attach the gradient from z_st without changing the forward value.
    delta = z_st - z_hard
    weights = weights_fwd + delta - delta.detach()

    if torch.is_grad_enabled():
        forward_diff = (weights.detach() - weights_fwd.detach()).abs().max()
        assert forward_diff < 1e-6, f"Continuous gating forward mismatch: max diff {forward_diff.item()}"

    return weights, z_st


def debug_check_continuous_gating(batch: int = 2, feat_dim: int = 8, top_m: int = 3, tau: float = 0.5):
    """Run a quick forward/backward sanity check for continuous Top-M gating."""

    logits = torch.randn(batch, feat_dim, requires_grad=True)
    w_soft = torch.sigmoid(logits / tau)
    z_hard = hard_topk_mask(logits, top_m)
    weights, _ = compose_continuous_topm_weights(z_hard, w_soft)

    loss = weights.sum()
    loss.backward()

    print(f"max|weights - z_hard*w_soft|: {(weights - z_hard * w_soft).abs().max().item():.6f}")
    print(f"logits grad norm: {logits.grad.norm().item():.6f}")
