"""Bounded NumPy evaluators for symbolic expressions.

``safe_evaluate`` implements protected search/reward semantics.
``ordinary_evaluate`` is the fail-closed real-mathematics endpoint used for
candidate selection and reporting.
"""

from typing import List, Optional

import numpy as np
import sympy as sp

from symba_grpo.gp.primitives import OVERFLOW_LIMIT

_safe_abs = sp.Function("_safe_abs")
_safe_pow = sp.Function("_safe_pow")


def _numpy_safe_pow(base, exponent):
    """Match GP protected-power semantics without constructing Python integers."""
    base = np.asarray(base, dtype=float)
    exponent = np.asarray(exponent, dtype=float)
    with np.errstate(all="ignore"):
        result = np.power(np.abs(base), exponent)
    return np.where(np.abs(result) < OVERFLOW_LIMIT, result, OVERFLOW_LIMIT)


def _protect_expr(expr: sp.Expr) -> sp.Expr:
    """Lower every power to protected floating-point evaluation.

    ``_safe_pow`` is an inert SymPy function mapped to NumPy at lambdify time.
    This prevents Python from constructing huge constant integers while still
    matching the GP primitive's absolute-base and overflow semantics.

    Only rebuilds tree nodes that actually changed to avoid expensive
    expr.func(*args) canonicalization on unmodified subtrees.
    """
    if not expr.args:
        return expr
    new_args = [_protect_expr(a) for a in expr.args]
    if isinstance(expr, sp.Pow):
        return _safe_pow(*new_args)
    changed = any(n is not o for n, o in zip(new_args, expr.args))
    if not changed:
        return expr
    rebuilt = expr.func(*new_args, evaluate=False)
    return rebuilt


PROTECTED_NUMPY_MODULES = [
    {
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(np.where(np.abs(x) > 1e-300, np.abs(x), 1e-300)),
        "exp": lambda x: np.exp(np.clip(x, -500, 500)),
        "Abs": np.abs,
        "_safe_abs": np.abs,
        "_safe_pow": _numpy_safe_pow,
        "asin": lambda x: np.arcsin(np.clip(x, -1, 1)),
        "acos": lambda x: np.arccos(np.clip(x, -1, 1)),
    },
    "numpy",
]

def safe_lambdify(
    var_symbols: List[sp.Symbol],
    expr: sp.Expr,
) -> callable:
    """Lambdify with protected numpy modules matching GP semantics."""
    return sp.lambdify(var_symbols, _protect_expr(expr), modules=PROTECTED_NUMPY_MODULES)


def _column_arguments(var_symbols, X):
    """Map canonical s_k symbols to their declared zero-based data columns."""
    arguments = []
    for symbol in var_symbols:
        name = str(symbol)
        if not name.startswith("s_") or not name[2:].isdigit():
            raise ValueError(f"unsupported variable {name!r}")
        column = int(name[2:]) - 1
        if column < 0 or column >= X.shape[1]:
            raise ValueError(f"{name} is outside the {X.shape[1]}-column support")
        arguments.append(X[:, column])
    return arguments


def safe_evaluate(
    expr: sp.Expr,
    X: np.ndarray,
    var_symbols: Optional[List[sp.Symbol]] = None,
) -> np.ndarray:
    """Evaluate a SymPy expression on data, returning y_pred with NaN for failures.

    Returns an array of the same length as X's first axis. Entries that
    cannot be computed (domain errors, overflows) are np.nan.
    """
    if var_symbols is None:
        var_symbols = sorted(
            [s for s in expr.free_symbols if str(s).startswith("s_")],
            key=str,
        )

    n_rows = X.shape[0]

    try:
        func = safe_lambdify(var_symbols, expr)
        args = _column_arguments(var_symbols, X)

        with np.errstate(all="ignore"):
            y_pred = np.array(func(*args), dtype=np.float64)

        if y_pred.ndim == 0:
            y_pred = np.full(n_rows, float(y_pred))
        if y_pred.shape != (n_rows,) or not np.isfinite(y_pred).all():
            return np.full(n_rows, np.nan)
        return y_pred
    except Exception:
        return np.full(n_rows, np.nan)


def ordinary_evaluate(
    expr: sp.Expr,
    X: np.ndarray,
    var_symbols: Optional[List[sp.Symbol]] = None,
) -> np.ndarray:
    """Boundedly evaluate a SymPy tree with ordinary real NumPy semantics."""
    n_rows = X.shape[0]
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        return np.full(n_rows, np.nan)
    allowed_symbols = None if var_symbols is None else {str(symbol) for symbol in var_symbols}
    node_budget = 1024

    def reciprocal(value):
        with np.errstate(all="ignore"):
            return np.divide(1.0, value)

    def reduce_values(function, values):
        result = values[0]
        for value in values[1:]:
            result = function(result, value)
        return result

    operations = {
        sp.Add: lambda *values: reduce_values(np.add, values),
        sp.Mul: lambda *values: reduce_values(np.multiply, values),
        sp.Pow: np.power,
        sp.exp: np.exp,
        sp.log: np.log,
        sp.Abs: np.abs,
        sp.sign: np.sign,
        sp.sin: np.sin,
        sp.cos: np.cos,
        sp.tan: np.tan,
        sp.cot: lambda value: reciprocal(np.tan(value)),
        sp.sec: lambda value: reciprocal(np.cos(value)),
        sp.csc: lambda value: reciprocal(np.sin(value)),
        sp.asin: np.arcsin,
        sp.acos: np.arccos,
        sp.atan: np.arctan,
        sp.acot: lambda value: np.arctan(reciprocal(value)),
        sp.asec: lambda value: np.arccos(reciprocal(value)),
        sp.acsc: lambda value: np.arcsin(reciprocal(value)),
        sp.sinh: np.sinh,
        sp.cosh: np.cosh,
        sp.tanh: np.tanh,
        sp.coth: lambda value: reciprocal(np.tanh(value)),
        sp.sech: lambda value: reciprocal(np.cosh(value)),
        sp.csch: lambda value: reciprocal(np.sinh(value)),
        sp.asinh: np.arcsinh,
        sp.acosh: np.arccosh,
        sp.atanh: np.arctanh,
        sp.acoth: lambda value: np.arctanh(reciprocal(value)),
        sp.asech: lambda value: np.arccosh(reciprocal(value)),
        sp.acsch: lambda value: np.arcsinh(reciprocal(value)),
    }

    def at(node: sp.Expr, depth: int = 0):
        nonlocal node_budget
        node_budget -= 1
        if node_budget < 0 or depth > 512:
            raise ValueError("ordinary-expression resource bound")
        if isinstance(node, sp.Symbol):
            name = str(node)
            if allowed_symbols is not None and name not in allowed_symbols:
                raise ValueError("undeclared expression symbol")
            if not name.startswith("s_") or not name[2:].isdigit():
                raise ValueError("unsupported expression symbol")
            column = int(name[2:]) - 1
            if not 0 <= column < X.shape[1]:
                raise ValueError("expression symbol outside support")
            return X[:, column]
        if node.is_Number:
            return float(node)
        if node == sp.pi:
            return float(np.pi)
        if node == sp.E:
            return float(np.e)
        arguments = [at(argument, depth + 1) for argument in node.args]
        operation = operations.get(node.func)
        if operation is None:
            raise ValueError("unsupported expression operation")
        with np.errstate(all="ignore"):
            result = operation(*arguments)
        if not np.isfinite(np.asarray(result, dtype=float)).all():
            raise ValueError("nonfinite ordinary-expression intermediate")
        return result

    try:
        with np.errstate(all="ignore"):
            prediction = np.asarray(at(expr), dtype=np.float64)
        if prediction.ndim == 0:
            prediction = np.full(n_rows, float(prediction))
        if prediction.shape != (n_rows,) or not np.isfinite(prediction).all():
            raise ValueError("invalid ordinary prediction")
        return prediction
    except Exception:
        return np.full(n_rows, np.nan)
