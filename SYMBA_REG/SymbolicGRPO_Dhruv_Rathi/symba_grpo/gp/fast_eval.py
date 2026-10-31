"""Vectorized NumPy fitness evaluation with protected GP operators."""

from __future__ import annotations

import numpy as np
from deap import gp

from symba_grpo.gp.primitives import OVERFLOW_LIMIT

_BIG = OVERFLOW_LIMIT
GP_SELECTION_TOP_N = 50
GP_SELECTION_R2_TOLERANCE = 1e-12


def _np_div(x1, x2):
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    with np.errstate(all="ignore"):
        out = np.where(x2 == 0.0, _BIG, x1 / np.where(x2 == 0.0, 1.0, x2))
    return out


def _np_exp(x):
    x = np.asarray(x, float)
    with np.errstate(all="ignore"):
        out = np.where(x > 100, _BIG, np.where(x < -100, 0.0, np.exp(np.clip(x, -100, 100))))
    return out


def _np_log(x):
    x = np.asarray(x, float)
    with np.errstate(all="ignore"):
        out = np.where(x != 0.0, np.log(np.abs(np.where(x == 0.0, 1.0, x))), -_BIG)
    return out


def _np_sqrt(x):
    with np.errstate(all="ignore"):
        return np.sqrt(np.abs(np.asarray(x, float)))


def _np_pow(x1, x2):
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    with np.errstate(all="ignore"):
        r = np.power(np.abs(x1), x2)
        return np.where(np.abs(r) < _BIG, r, _BIG)


# Names must match those DEAP renders in tree strings (see make_primitive_set / bridge).
def evaluate_deap_tree(individual: gp.PrimitiveTree, X: np.ndarray) -> np.ndarray:
    """Evaluate one GP tree structurally on a matrix without parsing SymPy.

    The traversal is linear in the actual tree.  At most one array per active
    tree level is retained, so a large but height-bounded GP individual cannot
    cause symbolic expansion at the reporting boundary.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(individual) == 0:
        return np.full(len(X) if X.ndim else 0, np.nan)

    operations = {
        "add": np.add, "sub": np.subtract, "mul": np.multiply,
        "protected_div": _np_div, "protected_pow": _np_pow,
        "protected_exp": _np_exp, "protected_log": _np_log,
        "protected_sqrt": _np_sqrt, "sin": np.sin, "cos": np.cos,
        "tan": np.tan, "tanh": np.tanh, "abs": np.abs,
    }

    def at(index: int):
        if index >= len(individual):
            raise ValueError("incomplete GP tree")
        node = individual[index]
        if isinstance(node, gp.Terminal):
            value = node.value
            if isinstance(value, str) and value.startswith("s_") and value[2:].isdigit():
                column = int(value[2:]) - 1
                if column < 0 or column >= X.shape[1]:
                    raise ValueError("undeclared variable")
                return X[:, column], index + 1
            if value == "pi":
                return np.pi, index + 1
            if value == "E":
                return np.e, index + 1
            if type(value) is int and 1 <= value <= 4:
                return float(value), index + 1
            raise ValueError("unsupported terminal")
        if not isinstance(node, gp.Primitive) or node.name not in operations:
            raise ValueError("unsupported primitive")
        args, next_index = [], index + 1
        for _ in range(node.arity):
            value, next_index = at(next_index)
            args.append(value)
        with np.errstate(all="ignore"):
            return operations[node.name](*args), next_index

    try:
        prediction, consumed = at(0)
        if consumed != len(individual):
            raise ValueError("trailing GP nodes")
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim == 0:
            prediction = np.full(X.shape[0], float(prediction))
        if prediction.shape != (X.shape[0],) or not np.isfinite(prediction).all():
            raise ValueError("invalid GP prediction")
        return prediction
    except Exception:
        return np.full(X.shape[0], np.nan)


def evaluate_deap_tree_ordinary(individual: gp.PrimitiveTree, X: np.ndarray) -> np.ndarray:
    """Evaluate a GP tree under ordinary real-valued mathematical semantics.

    Evolution deliberately uses protected operators.  Final candidate
    selection and reporting do not: division by zero, invalid real powers,
    nonpositive logarithms, negative square roots, and overflow fail closed.
    The traversal stays structural and linear in the tree, so it does not
    invoke SymPy or symbolic canonicalization.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(individual) == 0:
        return np.full(len(X) if X.ndim else 0, np.nan)

    operations = {
        "add": np.add, "sub": np.subtract, "mul": np.multiply,
        "protected_div": np.divide, "protected_pow": np.power,
        "protected_exp": np.exp, "protected_log": np.log,
        "protected_sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
        "tan": np.tan, "tanh": np.tanh, "abs": np.abs,
    }

    def at(index: int):
        if index >= len(individual):
            raise ValueError("incomplete GP tree")
        node = individual[index]
        if isinstance(node, gp.Terminal):
            value = node.value
            if isinstance(value, str) and value.startswith("s_") and value[2:].isdigit():
                column = int(value[2:]) - 1
                if column < 0 or column >= X.shape[1]:
                    raise ValueError("undeclared variable")
                return X[:, column], index + 1
            if value == "pi":
                return np.pi, index + 1
            if value == "E":
                return np.e, index + 1
            if type(value) is int and 1 <= value <= 4:
                return float(value), index + 1
            raise ValueError("unsupported terminal")
        if not isinstance(node, gp.Primitive) or node.name not in operations:
            raise ValueError("unsupported primitive")
        arguments, next_index = [], index + 1
        for _ in range(node.arity):
            argument, next_index = at(next_index)
            arguments.append(argument)
        with np.errstate(all="ignore"):
            result = operations[node.name](*arguments)
        if not np.isfinite(np.asarray(result, dtype=float)).all():
            raise ValueError("nonfinite ordinary-mathematics intermediate")
        return result, next_index

    try:
        prediction, consumed = at(0)
        if consumed != len(individual):
            raise ValueError("trailing GP nodes")
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim == 0:
            prediction = np.full(X.shape[0], float(prediction))
        if prediction.shape != (X.shape[0],) or not np.isfinite(prediction).all():
            raise ValueError("invalid ordinary-mathematics prediction")
        return prediction
    except Exception:
        return np.full(X.shape[0], np.nan)


def _fitness(individual: gp.PrimitiveTree) -> float:
    try:
        value = float(individual.fitness.values[0])
        return value if np.isfinite(value) else float("inf")
    except Exception:
        return float("inf")


def _fingerprint(individual: gp.PrimitiveTree) -> tuple:
    from symba_grpo.gp.bridge import deap_tree_fingerprint

    return deap_tree_fingerprint(individual)


def _canonical_sort_key(individual: gp.PrimitiveTree) -> tuple:
    """Return a type-stable lexical key for deterministic GP ordering."""
    return tuple(
        (kind, type(value).__name__, repr(value), int(arity))
        for kind, value, arity in _fingerprint(individual)
    )


def select_ordinary_gp_candidate(
    population: list,
    hall_of_fame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    top_n: int = GP_SELECTION_TOP_N,
    tie_tolerance: float = GP_SELECTION_R2_TOLERANCE,
):
    """Select a final GP candidate without using the target formula.

    The deterministic shortlist is the Hall of Fame plus the best protected-
    fitness members of the final population.  Candidates are rescored on the
    full policy panel with ordinary mathematical semantics.  Numerical ties
    are resolved by smaller trees and then a structural fingerprint.
    Returns ``(tree_or_none, r2, diagnostics)``.
    """
    from symba_grpo.evaluation.metrics import r2_score

    if top_n <= 0 or tie_tolerance < 0:
        raise ValueError("invalid GP final-selection configuration")
    ordered_population = sorted(
        population,
        key=lambda individual: (_fitness(individual), len(individual), _canonical_sort_key(individual)),
    )[:top_n]
    shortlist, seen = [], set()
    for individual in [*list(hall_of_fame or ()), *ordered_population]:
        key = _fingerprint(individual)
        if key not in seen:
            shortlist.append(individual)
            seen.add(key)

    scored = []
    for individual in shortlist:
        prediction = evaluate_deap_tree_ordinary(individual, X)
        score = r2_score(y, prediction)
        if np.isfinite(score):
            scored.append((individual, score, len(individual), _canonical_sort_key(individual)))
    diagnostics = {
        "shortlist_size": len(shortlist),
        "ordinary_valid": len(scored),
        "numerical_ties": 0,
    }
    if not scored:
        return None, -float("inf"), diagnostics
    best_score = max(item[1] for item in scored)
    tied = [item for item in scored if item[1] >= best_score - tie_tolerance]
    diagnostics["numerical_ties"] = len(tied)
    selected = min(tied, key=lambda item: (item[2], item[3]))
    return selected[0], selected[1], diagnostics


def make_vectorized_evaluator(points: list):
    """Return a DEAP-compatible `evaluate(individual) -> (mse,)` over `points`.

    `points` is the GP fitness set [(x_list, y), ...]; we stack it once into
    arrays and evaluate every individual on the whole batch.
    """
    X = np.asarray([p[0] for p in points], dtype=float)   # (N, n_vars)
    y = np.asarray([p[1] for p in points], dtype=float)    # (N,)
    n_vars = X.shape[1] if X.ndim == 2 else 1
    n = X.shape[0]
    fallback = float(np.mean(np.full(n, _BIG)))  # all-bad fitness == OVERFLOW_LIMIT

    def evaluate(individual):
        try:
            with np.errstate(all="ignore"):
                preds = evaluate_deap_tree(individual, X)
                preds = np.where(np.isfinite(preds), preds, _BIG)
                errs = (preds - y) ** 2
                errs = np.where(np.isfinite(errs), errs, _BIG)
                mse = float(np.mean(errs))
            return (mse if np.isfinite(mse) else fallback,)
        except Exception:
            return (fallback,)

    return evaluate
