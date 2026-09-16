"""JEPA alignment components: predictor MLP and cosine alignment loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """2-layer MLP projecting z_num into z_sym's space."""

    def __init__(self, d_model: int, d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, z_num: torch.Tensor) -> torch.Tensor:
        """(batch, d_model) -> (batch, d_model)"""
        return self.net(z_num)


class IdentityPredictor(nn.Module):
    """No-op predictor: z_pred = z_num."""

    def forward(self, z_num: torch.Tensor) -> torch.Tensor:
        return z_num


def jepa_loss(
    z_pred: torch.Tensor,
    z_sym: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """Cosine alignment loss: (1 - cosine_similarity).mean().

    Args:
        z_pred: (batch, d_model) from predictor(z_num).
        z_sym: (batch, d_model) — detach at call site if needed.
        mode: "cosine" (default) or "raw" — ordinary pairwise cosine;
              "centered" — subtracts batch mean before cosine.
    Returns:
        Scalar loss in [0, 2].
    """
    if mode == "centered":
        z_pred = z_pred - z_pred.mean(0)
        z_sym = z_sym - z_sym.mean(0)
    cos = F.cosine_similarity(z_pred, z_sym, dim=-1)
    return (1 - cos).mean()


def subsample_consistency_loss(
    z_views: list[torch.Tensor],
    mode: str = "cosine",
) -> torch.Tensor:
    """Mean cosine distance over all pairs of views of the same equations.

    Encourages the point-cloud encoder to map different numerical samples
    of one function to the same latent.  No predictor and no stop-gradient:
    every view contributes gradient.

    Args:
        z_views: list of (batch, d_model) encoder outputs, one per view.
            Entry i and entry j must describe the same equations in the
            same batch order.
        mode: ``"cosine"`` (or ``"raw"``) for ordinary pairwise cosine;
            ``"centered"`` to subtract each view's batch mean first.

    Returns:
        Scalar loss in [0, 2].  A single view gives 0.

    Note:
        The two modes are **not** on the same scale, so a given lambda does
        not mean the same thing in each.  The T-Net max-pools ReLU features
        into the positive orthant, so raw cosine between any two embeddings
        starts near 0.99 and the raw loss starts near zero — little gradient
        and easy to satisfy without learning anything.  Centering removes
        that shared offset, so the loss acts on the discriminative part of
        the representation and starts one to two orders of magnitude larger.

        Centering also changes the failure mode: under total collapse every
        centered vector is 0, cosine evaluates to 0, and the loss goes to 1.
        The centered objective therefore *penalises* collapse, whereas the
        raw objective is minimised by it.
    """
    n = len(z_views)
    if n < 2:
        # Nothing to align; keep the graph intact by returning a zero that
        # matches the incoming dtype/device when possible.
        if n == 1:
            return z_views[0].sum() * 0.0
        raise ValueError("subsample_consistency_loss needs at least one view")

    if mode == "centered":
        z_views = [z - z.mean(0) for z in z_views]
    elif mode not in ("cosine", "raw"):
        raise ValueError(
            f"unknown mode {mode!r}; expected 'cosine', 'raw', or 'centered'"
        )

    losses = [
        (1.0 - F.cosine_similarity(z_views[i], z_views[j], dim=-1)).mean()
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return torch.stack(losses).mean()
