"""JEPA diagnostics: spread, retrieval, common-mode, view consistency."""

import numpy as np
import torch
import torch.nn.functional as F


def _pairwise_cos(z: torch.Tensor) -> torch.Tensor:
    """Return (B, B) pairwise cosine similarity matrix."""
    z_norm = F.normalize(z.float(), dim=-1)
    return z_norm @ z_norm.T


def sym_spread(z_sym: torch.Tensor) -> dict:
    """Off-diagonal mean pairwise cosine, raw and mean-centered.

    High off-diag cos = collapsed / anisotropic embeddings.
    """
    B = z_sym.shape[0]
    # Raw
    cos_raw = _pairwise_cos(z_sym)
    mask = ~torch.eye(B, device=z_sym.device, dtype=torch.bool)
    raw_offdiag = cos_raw[mask].mean().item()

    # Centered
    z_cent = z_sym - z_sym.mean(0)
    cos_cent = _pairwise_cos(z_cent)
    cent_offdiag = cos_cent[mask].mean().item()

    return {"raw": raw_offdiag, "centered": cent_offdiag}


def pred_spread(z_pred: torch.Tensor) -> dict:
    """Off-diagonal mean pairwise cosine of predictor outputs.

    High value = predictor outputting near-constant vector.
    """
    B = z_pred.shape[0]
    cos = _pairwise_cos(z_pred)
    mask = ~torch.eye(B, device=z_pred.device, dtype=torch.bool)
    return {"raw": cos[mask].mean().item()}


def retrieval_top1(z_pred: torch.Tensor, z_sym: torch.Tensor) -> dict:
    """Row-argmax of cosine matrix == diagonal → retrieval accuracy.

    Reports raw and centered versions. Chance = 1/B.
    """
    B = z_pred.shape[0]

    # Raw
    zp_f, zs_f = z_pred.float(), z_sym.float()
    cos_raw = F.normalize(zp_f, dim=-1) @ F.normalize(zs_f, dim=-1).T
    acc_raw = (cos_raw.argmax(dim=1) == torch.arange(B, device=z_pred.device)).float().mean().item()

    # Centered
    zp = zp_f - zp_f.mean(0)
    zs = zs_f - zs_f.mean(0)
    cos_cent = F.normalize(zp, dim=-1) @ F.normalize(zs, dim=-1).T
    acc_cent = (cos_cent.argmax(dim=1) == torch.arange(B, device=z_pred.device)).float().mean().item()

    return {"raw": acc_raw, "centered": acc_cent, "chance": 1.0 / B}


def common_mode(z: torch.Tensor) -> dict:
    """Common-mode analysis: mean-norm ratio and top-3 PC variance share.

    mean_norm_ratio = ||mean(z)|| / mean(||z||).  Close to 1 = strong common mode.
    pc_var_share = fraction of variance in top-3 PCs (via pca_lowrank).
    """
    z = z.float()
    mean_vec = z.mean(0)
    mean_norm = mean_vec.norm().item()
    avg_norm = z.norm(dim=-1).mean().item()
    ratio = mean_norm / (avg_norm + 1e-10)

    # Top-3 PC variance share
    z_centered = z - mean_vec
    try:
        _, S, _ = torch.pca_lowrank(z_centered, q=min(3, z.shape[0], z.shape[1]))
        var_explained = S ** 2
        total_var = z_centered.var(dim=0).sum() * z.shape[0]
        pc_share = (var_explained.sum() / (total_var + 1e-10)).item()
    except Exception:
        pc_share = float("nan")

    return {"mean_norm_ratio": ratio, "pc_top3_var_share": pc_share}


# ---------------------------------------------------------------------------
# Subsample-JEPA: view consistency
# ---------------------------------------------------------------------------

def view_consistency_diagnostics(z_all: torch.Tensor, n_views: int) -> dict:
    """Same-function vs different-function cosine similarity.

    Args:
        z_all: (n_exprs * n_views, d_model) encoder outputs, with the views
            of each expression contiguous:
            ``[e0v0, e0v1, ..., e1v0, e1v1, ...]``.
        n_views: number of views per expression.

    Returns:
        dict with:
          same_fn_cos  — mean cosine between distinct views of one equation
                         (invariance to point resampling; want it high)
          diff_fn_cos  — mean cosine between views of different equations
                         (global collapse indicator; want it low)
          gap          — same_fn_cos - diff_fn_cos, the headline metric

        plus ``*_centered`` variants computed after removing the batch mean.
        The T-Net max-pools ReLU activations, so raw embeddings sit in the
        positive orthant and *every* pair scores ~0.98 regardless of content.
        The centered numbers strip that shared offset and are the ones worth
        reading; the raw ones are kept for continuity with sym_spread.
    """
    n_total = z_all.shape[0]
    nan = float("nan")
    if n_views < 2 or n_total < n_views * 2:
        return {"same_fn_cos": nan, "diff_fn_cos": nan, "gap": nan,
                "same_fn_cos_centered": nan, "diff_fn_cos_centered": nan,
                "gap_centered": nan}

    n_exprs = n_total // n_views
    z_f = z_all.float()

    group = torch.arange(n_exprs, device=z_f.device).repeat_interleave(n_views)
    same = group[:, None] == group[None, :]
    eye = torch.eye(n_total, dtype=torch.bool, device=z_f.device)
    same_off_diag, diff = same & ~eye, ~same

    def _stats(z):
        cos = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).T
        return cos[same_off_diag].mean().item(), cos[diff].mean().item()

    same_cos, diff_cos = _stats(z_f)
    same_c, diff_c = _stats(z_f - z_f.mean(0))

    return {
        "same_fn_cos": same_cos,
        "diff_fn_cos": diff_cos,
        "gap": same_cos - diff_cos,
        "same_fn_cos_centered": same_c,
        "diff_fn_cos_centered": diff_c,
        "gap_centered": same_c - diff_c,
    }


@torch.no_grad()
def generate_diagnostic_embeddings(
    dataset,
    encoder,
    n_exprs: int,
    n_views: int,
    eval_seed: int,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Encode *n_views* fixed views of the first *n_exprs* dataset equations.

    View seeds come from ``(eval_seed, 0, expr_idx, view_idx)``, so the same
    clouds are used for every checkpoint and every model — differences in the
    reported metrics reflect the encoder, not the data.

    Returns:
        (n_exprs * n_views, d_model) tensor, views of each expression
        contiguous — the layout ``view_consistency_diagnostics`` expects.
    """
    from symbolic_jepa.dataset import (
        MultiViewPointCloudDataset, sample_and_normalize,
    )

    n_exprs = min(n_exprs, len(dataset))
    clouds = []
    for expr_idx in range(n_exprs):
        expr = dataset.samples[expr_idx]['expr']
        for v in range(n_views):
            seed = MultiViewPointCloudDataset.view_seed(eval_seed, 0, expr_idx, v)
            clouds.append(sample_and_normalize(
                expr, dataset.n_points, dataset.target_d,
                rng=np.random.RandomState(seed), random_subsample=True,
            ))

    # Encode in chunks: the T-Net expands every point to 4*d_model, so a
    # single forward over all views would be far larger than a train batch.
    was_training = encoder.training
    encoder.eval()
    try:
        out = [
            encoder(torch.stack(clouds[i:i + batch_size]).to(device))
            for i in range(0, len(clouds), batch_size)
        ]
    finally:
        encoder.train(was_training)

    return torch.cat(out, dim=0)
