"""Evaluation metrics for symbolic regression.

Includes numeric, token-level, symbolic equivalence, and teacher-forced measures.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

import numpy as np
import sympy as sp

EXACT_SYMBOLIC_MATCH_TIMEOUT_S = 5.0
EXACT_STATUSES = {
    "exact", "nonmatch", "parse_failure", "worker_failure", "resource_limit", "wall_timeout",
}


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        return float("inf")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        return -float("inf")
    yt, yp = y_true, y_pred
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res < 1e-10 else 0.0
    return float(1.0 - ss_res / ss_tot)


def normalised_mse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Normalised MSE: MSE divided by variance of y.

    NMSE = mean((y - y_hat)^2) / var(y)
    Returns 0 for perfect fit, 1 for predicting-the-mean performance.
    """
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        return float("inf")
    yt, yp = y_true, y_pred
    var_y = np.var(yt)
    if var_y < eps:
        return 0.0 if np.mean((yt - yp) ** 2) < eps else float("inf")
    return float(np.mean((yt - yp) ** 2) / var_y)


def token_accuracy(pred_tokens: List[str], true_tokens: List[str]) -> float:
    """Per-token accuracy between two token lists."""
    if not true_tokens:
        return 1.0 if not pred_tokens else 0.0
    matches = sum(1 for p, t in zip(pred_tokens, true_tokens) if p == t)
    return matches / max(len(true_tokens), 1)


def prediction_record(pred_expr) -> dict:
    """Serialize a prediction into the exact worker's replayable input schema."""
    from symba_grpo.gp.bridge import GPExpression

    if isinstance(pred_expr, GPExpression):
        return pred_expr.exact_worker_record()
    from symba_grpo.tokenizers.prefix import sympy_to_prefix
    return {"version": 1, "source": "beam", "tokens": sympy_to_prefix(pred_expr)}


def exact_symbolic_status_record(record: dict, true_expr: sp.Expr, timeout: float) -> str:
    """Run exact scoring and return a fail-closed, auditable status."""
    if timeout <= 0:
        return "wall_timeout"
    try:
        payload = json.dumps({
            "prediction": record,
            "truth": str(true_expr),
        })
    except Exception:
        return "parse_failure"
    worker = Path(__file__).with_name("exact_worker.py")
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "worker_failure"
    try:
        stdout, _ = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
        try:
            process.wait()
        except OSError:
            pass
        return "wall_timeout"
    if process.returncode != 0:
        return "resource_limit" if process.returncode < 0 else "worker_failure"
    try:
        response = json.loads(stdout)
        status = response.get("status")
        if status not in EXACT_STATUSES - {"wall_timeout"}:
            return "worker_failure"
        if bool(response.get("exact", False)) != (status == "exact"):
            return "worker_failure"
        return status
    except Exception:
        return "worker_failure"


def exact_symbolic_match_record(record: dict, true_expr: sp.Expr, timeout: float) -> bool:
    """Return whether resource-bounded exact scoring produced an exact match."""
    return exact_symbolic_status_record(record, true_expr, timeout) == "exact"


def exact_symbolic_match(
    pred_expr: sp.Expr,
    true_expr: sp.Expr,
    timeout: float = EXACT_SYMBOLIC_MATCH_TIMEOUT_S,
) -> bool:
    """Check exact equivalence in a killable, wall-time-bounded process."""
    return exact_symbolic_match_record(prediction_record(pred_expr), true_expr, timeout)


EVALUATE_EXPRESSION_TIMEOUT_S = 25.0
FUNCTIONAL_R2_THRESHOLD = 0.99


def evaluate_expression(
    pred_expr: sp.Expr,
    true_expr: sp.Expr,
    X: np.ndarray,
    y_true: np.ndarray,
    pred_tokens: Optional[List[str]] = None,
    true_tokens: Optional[List[str]] = None,
    timeout: float = EVALUATE_EXPRESSION_TIMEOUT_S,
) -> dict:
    """Compute all metrics for a single predicted expression.

    GP predictions remain structural for numerical scoring.  Exact symbolic
    work for every prediction runs in a killable, resource-limited process.
    """
    from symba_grpo.evaluation.safe_eval import ordinary_evaluate
    from symba_grpo.gp.bridge import GPExpression
    from symba_grpo.gp.fast_eval import evaluate_deap_tree_ordinary

    if isinstance(pred_expr, GPExpression):
        y_pred = evaluate_deap_tree_ordinary(pred_expr.tree, X)
    else:
        var_symbols = [
            symbol for symbol in sorted(pred_expr.free_symbols, key=str)
            if str(symbol).startswith("s_")
        ]
        y_pred = ordinary_evaluate(pred_expr, X, var_symbols)
    numerical_r2 = r2_score(y_true, y_pred)
    exact_status = exact_symbolic_status_record(
        prediction_record(pred_expr), true_expr,
        timeout=min(timeout, EXACT_SYMBOLIC_MATCH_TIMEOUT_S),
    )
    exact = exact_status == "exact"
    results = {
        "r2": numerical_r2,
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mse_n": normalised_mse(y_true, y_pred),
        "exact_match": exact,
        "exact_status": exact_status,
        "functional_score": bool(
            np.isfinite(numerical_r2) and numerical_r2 >= FUNCTIONAL_R2_THRESHOLD
        ),
        "eval_timeout": exact_status == "wall_timeout",
    }

    if pred_tokens is not None and true_tokens is not None:
        results["token_accuracy"] = token_accuracy(pred_tokens, true_tokens)

    return results


def evaluate_batch(results: List[dict]) -> dict:
    """Aggregate metrics across multiple equations."""
    if not results:
        return {}

    agg = {}
    numeric_keys = ["r2", "mse", "rmse", "mse_n", "token_accuracy"]
    bool_keys = ["exact_match", "functional_score"]

    for key in numeric_keys:
        vals = [r[key] for r in results if r.get(key) is not None and np.isfinite(r[key])]
        agg[f"avg_{key}"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"median_{key}"] = float(np.median(vals)) if vals else float("nan")

    for key in bool_keys:
        vals = [r[key] for r in results if key in r]
        agg[f"{key}_rate"] = float(np.mean(vals)) if vals else 0.0
        agg[f"{key}_count"] = sum(vals)

    exact_statuses = [result.get("exact_status", "not_scored") for result in results]
    agg["exact_status_counts"] = {
        status: exact_statuses.count(status) for status in sorted(set(exact_statuses))
    }

    agg["total_equations"] = len(results)
    return agg
