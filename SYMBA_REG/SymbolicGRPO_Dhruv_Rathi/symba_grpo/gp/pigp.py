"""Transformer candidate scoring and conversion for PIGP."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import sympy as sp

from symba_grpo.evaluation.metrics import r2_score
from symba_grpo.evaluation.safe_eval import ordinary_evaluate


def _safe_r2(expression: sp.Expr, X: np.ndarray, y: np.ndarray) -> float:
    prediction = ordinary_evaluate(expression, X)
    return r2_score(y, prediction)


def _score_candidates(
    expressions: List[sp.Expr],
    X: np.ndarray,
    y: np.ndarray,
    support_n: int = 500,
    support_indices: np.ndarray | None = None,
) -> List[Tuple[sp.Expr, float]]:
    """Rank one fixed-constant candidate set on a shared numerical support."""
    if support_indices is None:
        support_size = min(support_n, len(X))
        indices = np.random.choice(len(X), support_size, replace=False)
    else:
        indices = np.asarray(support_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(X):
            raise ValueError("invalid explicit candidate-scoring support")
    support_X, support_y = X[indices], y[indices]
    scored = []
    for expression in expressions:
        try:
            score = r2_score(support_y, ordinary_evaluate(expression, support_X))
            if np.isfinite(score):
                scored.append((expression, score))
        except Exception:
            continue
    return sorted(scored, key=lambda item: item[1], reverse=True)
