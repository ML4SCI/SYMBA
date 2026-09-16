"""
Evaluation utilities for symbolic regression.

Includes constant fitting (BFGS), R² scoring, token accuracy,
and algebraic equivalence checking.
"""

import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize
from tqdm.auto import tqdm

from symbolic_jepa.tokenizer import prefix_to_sympy


def cleanup_eval_pool():
    """No-op — kept for backward compatibility with notebook code."""
    pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    return 1 - ss_res / (ss_tot + 1e-10)


def build_prefix_tree(token_sequences, progress: bool = False) -> dict:
    """Map each token prefix to its number of distinct continuations.

    Fit on the train sequences and pass to teacher_forced_counts to score only
    the positions where more than one token is possible.

    Args:
        token_sequences: iterable of token-ID sequences (e.g. dataset.token_keys).
        progress: Show a tqdm bar. Worth it above ~50k sequences: the loop is
            O(total tokens) prefix tuples.
    """
    counts: dict = {}
    if progress:
        token_sequences = tqdm(token_sequences, desc='building prefix tree',
                               leave=False)
    for seq in token_sequences:
        seq = tuple(int(t) for t in seq)
        for i in range(1, len(seq)):
            counts.setdefault(seq[:i], set()).add(seq[i])
    return {ctx: len(nxt) for ctx, nxt in counts.items()}


def branching_mask(targets, tree: dict, pad_id: int,
                   unseen_is_branching: bool = True):
    """(batch, seq-1) bool mask, True where more than one continuation is possible.

    Args:
        targets: (batch, seq) — ground-truth token IDs.
        tree: build_prefix_tree output.
        pad_id: Token ID used for padding.
        unseen_is_branching: How to treat a prefix absent from *tree*.
    """
    rows = []
    for row in targets.tolist():
        ctx, flags = [row[0]], []
        for t in row[1:]:
            if t == pad_id:
                flags.append(False)
            else:
                n = tree.get(tuple(ctx))
                flags.append(unseen_is_branching if n is None else n > 1)
            ctx.append(t)
        rows.append(flags)
    return torch.tensor(rows, dtype=torch.bool, device=targets.device)


def teacher_forced_accuracy(logits, targets, pad_id: int,
                            branch_tree: dict | None = None) -> float:
    """Fraction of non-pad positions where argmax matches target.

    Excludes the trivial data-token → <sos> prediction (position 0).

    Args:
        logits: (batch, 1+seq, vocab) — includes data-token position.
        targets: (batch, seq) — ground-truth token IDs.
        pad_id: Token ID used for padding.
        branch_tree: Optional build_prefix_tree output; restricts scoring to
            positions with more than one possible continuation.
    """
    correct, total = teacher_forced_counts(logits, targets, pad_id, branch_tree)
    return correct / (total + 1e-10)


def teacher_forced_counts(logits, targets, pad_id: int,
                          branch_tree: dict | None = None) -> tuple[float, float]:
    """Return (n_correct, n_total) for token-level accuracy.

    Same logic as teacher_forced_accuracy but returns raw counts
    for proper aggregation across batches of different lengths.
    """
    pred = logits[:, 1:-1, :].argmax(dim=-1)   # (batch, seq-1)
    tgt = targets[:, 1:]                        # (batch, seq-1)
    mask = (tgt != pad_id)
    if branch_tree is not None:
        mask = mask & branching_mask(targets, branch_tree, pad_id)
    correct = ((pred == tgt) & mask).float().sum().item()
    total = mask.float().sum().item()
    return correct, total


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def fit_constants(expr, constants, X, Y, var_syms, maxiter=100):
    """Fit fittable constants in a predicted expression using L-BFGS-B.

    Args:
        expr: SymPy expression (may contain c_0, c_1, ... symbols).
        constants: List of SymPy Symbol objects for fittable constants.
        X: (n_points, n_vars) input data.
        Y: (n_points,) target output.
        var_syms: List of SymPy Symbols for input variables.
        maxiter: Maximum BFGS iterations.

    Returns:
        (fitted_dict, Y_pred, r2) or (None, None, -inf) on failure.
    """
    if len(constants) == 0:
        f = sp.lambdify(var_syms, expr, 'numpy')
        try:
            Y_pred = np.broadcast_to(
                np.asarray(f(*X.T), dtype=float), Y.shape
            ).copy()
            return {}, Y_pred, r2_score(Y, Y_pred)
        except Exception:
            return None, None, -np.inf

    f = sp.lambdify(list(var_syms) + list(constants), expr, 'numpy')

    def loss(c):
        with np.errstate(all='ignore'):
            try:
                p = np.asarray(f(*X.T, *c), dtype=float)
                return float(np.mean((p - Y) ** 2)) if np.all(np.isfinite(p)) else 1e10
            except Exception:
                return 1e10

    r = minimize(loss, np.ones(len(constants)), method='L-BFGS-B',
                 options={'maxiter': maxiter})

    if not np.isfinite(r.fun) or r.fun >= 1e9:
        return None, None, -np.inf

    fitted = dict(zip([str(c) for c in constants], r.x))
    with np.errstate(all='ignore'):
        Y_pred = np.asarray(f(*X.T, *r.x), dtype=float)
    if not np.all(np.isfinite(Y_pred)):
        return None, None, -np.inf
    return fitted, Y_pred, r2_score(Y, Y_pred)


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------

def equations_equivalent(pred_str: str, gt_str: str, timeout: int = 2) -> bool:
    """Check if two prefix strings are algebraically equivalent.

    Uses only expand on the difference — no .equals, no sp.simplify — so it
    cannot hang on pathological expressions.  Handles commutativity (SymPy
    normalizes) and constant permutation.

    Args:
        timeout: Unused, kept for call-site compatibility.  The checks are
            bounded by construction, so there is nothing to time out.
    """
    try:
        pred_expr, pred_consts = prefix_to_sympy(pred_str)
        gt_expr, gt_consts = prefix_to_sympy(gt_str)
    except Exception:
        return False

    return _exprs_equivalent(pred_expr, pred_consts, gt_expr, gt_consts)


def _expand_equal(a, b) -> bool:
    """True if expand(a - b) collapses to zero.  Milliseconds, never hangs."""
    try:
        diff = sp.expand(a - b)
        return diff is sp.S.Zero or diff == 0
    except Exception:
        return False


def _exprs_equivalent(pred_expr, pred_consts, gt_expr, gt_consts) -> bool:
    """Equivalence check on already-parsed expressions.

    Deliberately avoids sp.Expr.equals(): it calls simplify() internally and
    runs unbounded on the deep expressions greedy decoding produces (measured
    30s+ on a single pair, vs 0.000s for expand).  On this vocabulary it
    catches nothing expand misses; genuine algebraic equivalence that expand
    cannot see is caught by the held-out R² check in evaluate_predictions.

    Note for anyone tempted to reinstate .equals() behind a signal timeout:
    the except-Exception clauses here swallow any Exception-derived alarm, so
    such a guard must derive from BaseException to work at all.
    """
    # Fast path: expand on the difference (handles commutativity, since
    # SymPy normalizes, plus most algebraic identities).
    if _expand_equal(pred_expr, gt_expr):
        return True

    # Constant permutation: if same number of constants, try all
    # permutations of pred constants to match gt constants.
    n_pred, n_gt = len(pred_consts), len(gt_consts)
    if n_pred == n_gt and 0 < n_pred <= 4:
        from itertools import permutations
        for perm in permutations(pred_consts):
            sub = dict(zip(perm, gt_consts))
            # simultaneous=True is required: subs applies a dict sequentially,
            # so a swap {c_0: c_1, c_1: c_0} would collapse both onto c_0.
            if _expand_equal(pred_expr.subs(sub, simultaneous=True), gt_expr):
                return True

    return False


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

# Deterministic seeds for R² evaluation, independent of model seed.
# Two separate seeds so fit and eval clouds are independent.
_R2_FIT_SEED  = 777_777_773
_R2_EVAL_SEED = 555_555_557


def evaluate_predictions(
    predictions: list[tuple[str, str]],
    dataset,
    tokenizer,
    n_fit_points: int = 200,
    n_eval_points: int = 200,
    r2_equiv_thresh: float = 0.999,
) -> dict:
    """Evaluate a list of (gt_prefix, pred_prefix) pairs.

    Equivalence is determined by held-out R² after constant fitting:
    constants are fitted on one deterministic point cloud, then R² is
    computed on a separate held-out cloud.  Both are seeded per-sample
    for reproducibility.

    ``predictions[i]`` must correspond to ``dataset.samples[i]``, so the eval
    loader has to be built with ``shuffle=False``.

    Args:
        predictions: List of (ground_truth_prefix, predicted_prefix) tuples.
        dataset: PointCloudDataset (for sampling evaluation points).
        tokenizer: PrefixTokenizer.
        n_fit_points: Number of points for constant fitting.
        n_eval_points: Number of held-out points for R² evaluation.
        r2_equiv_thresh: R² threshold for declaring equivalence (default 0.999).

    Returns:
        Dict with 'exact_match', 'token_accuracy', 'algebraic_equiv',
        'mean_r2', 'r2_above_0.9', and per-sample 'details'.
    """
    if len(predictions) > len(dataset.samples):
        raise ValueError(
            f'{len(predictions)} predictions but dataset has '
            f'{len(dataset.samples)} samples; ordering cannot be trusted.'
        )

    exact_matches = []
    token_accs = []
    equiv_matches = []
    r2_all = []  # one entry per prediction (None if not computable)
    r2_statuses = []  # parallel to r2_all: why each None happened
    details = []

    pbar = tqdm(predictions, desc='evaluate', leave=True)
    for i, (gt_str, pred_str) in enumerate(pbar):
        # Exact match
        exact = int(pred_str.strip() == gt_str.strip())
        exact_matches.append(exact)

        # Token accuracy
        pred_tokens = pred_str.split()
        gt_tokens = gt_str.split()
        denom = max(len(pred_tokens), len(gt_tokens))
        if denom == 0:
            token_accs.append(1.0)
        else:
            hits = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
            token_accs.append(hits / denom)

        # Parse both expressions
        parseable = False
        pred_expr = None
        pred_consts = None
        gt_expr = None
        gt_consts = None
        try:
            pred_expr, pred_consts = prefix_to_sympy(pred_str)
            parseable = True
        except Exception:
            pass

        try:
            gt_expr, gt_consts = prefix_to_sympy(gt_str)
        except Exception:
            pass

        # ── Equivalence cascade: fast sympy first, then R² ──
        equiv = exact  # start with exact match

        # Level 1: .equals, expand on difference, constant permutation
        # (milliseconds, no risk of hanging)
        if not equiv and pred_expr is not None and gt_expr is not None:
            equiv = int(_exprs_equivalent(
                pred_expr, pred_consts, gt_expr, gt_consts))

        # Level 2: R² via constant fitting on fit cloud, evaluated on held-out cloud.
        # Uses deterministic RNGs seeded per-sample for reproducibility.
        #
        # r2_status records WHY r2 is None.  'R²=N/A' collapses six unrelated
        # failures into one label, and they call for different fixes: a failed
        # constant fit is an optimiser problem, a too-small cloud is a sampling
        # problem, an unparseable prediction is a model problem.
        r2 = None
        r2_status = 'ok'
        if not parseable:
            r2_status = 'pred_unparseable'
        elif i >= len(dataset.samples):
            r2_status = 'no_dataset_sample'
        else:
            try:
                expr_obj = dataset.samples[i]['expr']
                n_vars = len(expr_obj.variables)
                var_syms = [sp.Symbol(f'x{j+1}') for j in range(n_vars)]

                # Fit cloud: fit constants
                fit_rng = np.random.RandomState(_R2_FIT_SEED + i)
                fit_cloud = expr_obj.sample(n_fit_points, rng=fit_rng)
                fit_mask = np.isfinite(fit_cloud).all(axis=1)
                fit_cloud = fit_cloud[fit_mask]

                if len(fit_cloud) < 50:
                    r2_status = f'fit_cloud_too_small({len(fit_cloud)}/{n_fit_points})'
                else:
                    X_fit = fit_cloud[:, :n_vars]
                    Y_fit = fit_cloud[:, n_vars]
                    fitted_dict, _, _ = fit_constants(
                        pred_expr, pred_consts, X_fit, Y_fit, var_syms,
                    )

                    if fitted_dict is None:
                        # L-BFGS-B never found a finite loss.  fit_constants
                        # scores a candidate 1e10 if ANY fit point is
                        # non-finite, so one domain violation (sqrt of a
                        # negative, division by ~0, a non-integer power of a
                        # negative x) flattens the objective and the fit dies.
                        r2_status = f'constant_fit_failed(n_consts={len(pred_consts)})'
                    else:
                        # Substitute fitted constants into expression
                        subs = {c: fitted_dict[str(c)] for c in pred_consts
                                if str(c) in fitted_dict}
                        fitted_expr = pred_expr.subs(subs) if subs else pred_expr
                        f_eval = sp.lambdify(var_syms, fitted_expr, 'numpy')

                        # Eval cloud: compute held-out R²
                        eval_rng = np.random.RandomState(_R2_EVAL_SEED + i)
                        eval_cloud = expr_obj.sample(n_eval_points, rng=eval_rng)
                        eval_mask = np.isfinite(eval_cloud).all(axis=1)
                        eval_cloud = eval_cloud[eval_mask]

                        if len(eval_cloud) < 50:
                            r2_status = (f'eval_cloud_too_small'
                                         f'({len(eval_cloud)}/{n_eval_points})')
                        else:
                            X_eval = eval_cloud[:, :n_vars]
                            Y_eval = eval_cloud[:, n_vars]
                            with np.errstate(all='ignore'):
                                Y_pred = np.broadcast_to(
                                    np.asarray(f_eval(*X_eval.T), dtype=float),
                                    Y_eval.shape,
                                ).copy()
                            n_bad = int((~np.isfinite(Y_pred)).sum())
                            if n_bad:
                                r2_status = (f'nonfinite_predictions'
                                             f'({n_bad}/{len(Y_pred)})')
                            else:
                                r2 = r2_score(Y_eval, Y_pred)
            except Exception as exc:
                r2_status = f'error:{type(exc).__name__}: {exc}'

        r2_all.append(r2)
        r2_statuses.append(r2_status)

        if r2 is not None and np.isfinite(r2):
            if not equiv and r2 >= r2_equiv_thresh:
                equiv = 1

        equiv_matches.append(equiv)

        details.append({
            'gt': gt_str, 'pred': pred_str,
            'exact': exact, 'equiv': equiv, 'r2': r2, 'parseable': parseable,
            'r2_status': r2_status,
        })

        # Update progress bar with running stats
        if (i + 1) % 10 == 0 or i == len(predictions) - 1:
            em = np.mean(exact_matches) * 100
            eq = np.mean(equiv_matches) * 100
            pbar.set_postfix(exact=f'{em:.0f}%', equiv=f'{eq:.0f}%')

    n = len(predictions)
    finite_r2 = [r for r in r2_all if r is not None and np.isfinite(r)]

    # Collapse the parameterised statuses to their kind for the summary, so
    # 'constant_fit_failed(n_consts=9)' and '(n_consts=12)' count together.
    from collections import Counter
    r2_status_counts = Counter(s.split('(')[0].split(':')[0]
                               for s in r2_statuses)

    results = {
        'exact_match': np.mean(exact_matches) if exact_matches else 0,
        'token_accuracy': np.mean(token_accs) if token_accs else 0,
        'algebraic_equiv': np.mean(equiv_matches) if equiv_matches else 0,
        'mean_r2': np.mean(finite_r2) if finite_r2 else float('nan'),
        'median_r2': np.median(finite_r2) if finite_r2 else float('nan'),
        # R²>0.9 out of ALL predictions (not just parseable)
        'r2_above_0.9': sum(1 for r in r2_all if r is not None and r > 0.9) / max(n, 1),
        'n_parseable': sum(1 for d in details if d['parseable']),
        'n_r2_computed': len(finite_r2),
        # Why R² was not computable, by kind. Sums to n_total.
        'r2_status_counts': dict(r2_status_counts.most_common()),
        'n_total': n,
        'details': details,
    }
    return results
