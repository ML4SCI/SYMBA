"""
Unified Expression class for SYMBA symbolic regression.

Provides a single representation for both Feynman equations and synthetic
expressions, using SymPy as the canonical internal form.
"""

import math
import pickle
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import sympy as sp
import sympy.assumptions  # force lazy submodule load (fixes Python 3.12)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor, implicit_application,
)

from symbolic_jepa.tokenizer import PrefixTokenizer, sympy_to_prefix, prefix_to_sympy


# ============================================================================
# VarMeta
# ============================================================================

@dataclass
class VarMeta:
    """Metadata for a single variable: original name and sampling bounds."""
    name: str
    low: float
    high: float


# ============================================================================
# Expression class
# ============================================================================

class Expression:
    """Unified symbolic expression for SYMBA.

    Wraps a SymPy expression with variable metadata.  Provides lazy
    prefix tokenization, evaluation via lambdify, and pluggable sampling.
    """

    def __init__(self, sympy_expr: sp.Expr, variables: list[VarMeta]):
        self._expr = sympy_expr
        self._variables = list(variables)
        # Lazy caches
        self._var_map: Optional[dict[str, str]] = None
        self._prefix: Optional[str] = None
        self._fn = None  # cached lambdify callable

    # --- Pickling ---

    def __getstate__(self):
        """Drop the lambdify cache: it is derived, and it is not picklable.

        ``sp.lambdify`` returns a function built by ``exec`` under a generated
        name, so pickling an Expression that has been sampled would otherwise
        fail with "Can't pickle <function _lambdifygenerated>".  It is rebuilt
        on the next ``evaluate()``.
        """
        state = self.__dict__.copy()
        state['_fn'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fn = None

    # --- Constructors ---

    @classmethod
    def from_sympy(cls, expr: sp.Expr, variables: list[VarMeta]) -> 'Expression':
        return cls(expr, variables)

    @classmethod
    def from_infix(cls, infix_str: str,
                   variables: Optional[list[VarMeta]] = None,
                   namespace: Optional[dict] = None) -> 'Expression':
        """Parse an infix string (e.g. "sin(x) + cos(2*x)") into an Expression.

        If variables is None, auto-detects free symbols and assigns
        default bounds [-pi, pi].
        """
        infix_str = infix_str.replace('^', '**')
        ns = namespace or {}
        expr = sp.sympify(infix_str, locals=ns)

        if variables is None:
            # Auto-detect: sort free symbols alphabetically for stable ordering
            free = sorted(expr.free_symbols, key=str)
            variables = [
                VarMeta(name=str(s), low=-math.pi, high=math.pi)
                for s in free
            ]

        return cls(expr, variables)

    @classmethod
    def from_feynman_row(cls, row: pd.Series) -> 'Expression':
        """Construct from a row of the Feynman CSV dataset.

        Extracts formula, variable names, and bounds from the row.
        Uses SymPy parsing with the Feynman symbol namespace.
        """
        formula = str(row['Formula']).replace('^', '**')
        formula_cleaned = _clean_formula_string(formula)

        # Build variable list from CSV columns
        variables = []
        for i in range(1, 11):
            name = row.get(f'v{i}_name')
            if pd.isna(name) or not isinstance(name, str) or not name.strip():
                continue
            low = float(row.get(f'v{i}_low', 1))
            high = float(row.get(f'v{i}_high', 5))
            variables.append(VarMeta(name=name.strip(), low=low, high=high))

        # Parse with Feynman namespace
        try:
            expr = parse_expr(
                formula_cleaned,
                transformations=_TRANSFORMATIONS,
                local_dict=_FEYNMAN_NAMESPACE,
                evaluate=False,
            )
        except Exception as e:
            raise ValueError(f"Cannot parse '{formula}': {e}") from e

        return cls(expr, variables)

    # --- Properties ---

    @property
    def sympy_expr(self) -> sp.Expr:
        return self._expr

    @property
    def variables(self) -> list[VarMeta]:
        return self._variables

    @property
    def var_map(self) -> dict[str, str]:
        """Mapping from original variable names to numbered tokens."""
        if self._var_map is None:
            self._var_map = {
                v.name: f'x{i+1}' for i, v in enumerate(self._variables)
            }
        return self._var_map

    @property
    def prefix(self) -> str:
        """Space-separated prefix-notation string."""
        if self._prefix is None:
            p = sympy_to_prefix(self._expr, self.var_map)
            if p is None:
                raise ValueError(
                    f"Cannot convert to prefix: {self._expr}"
                )
            self._prefix = p
        return self._prefix

    # --- Methods ---

    def tokenize(self, tokenizer: PrefixTokenizer) -> list[int]:
        """Encode prefix string to token IDs."""
        return tokenizer.encode(self.prefix)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate expression at given points.

        Args:
            points: (n_points, n_vars) array of input values.

        Returns:
            (n_points,) array of output values.
        """
        if self._fn is None:
            var_symbols = [sp.Symbol(v.name) for v in self._variables]
            self._fn = sp.lambdify(var_symbols, self._expr, 'numpy')

        result = self._fn(*points.T)
        return np.broadcast_to(np.asarray(result, dtype=float), (points.shape[0],)).copy()

    def sample(self, n_points: int = 200,
               method: str = 'uniform',
               rng: np.random.RandomState | None = None) -> np.ndarray:
        """Sample a point cloud from this expression.

        Args:
            n_points: Number of points to sample.
            method: 'uniform' (random within bounds),
                    'grid' (linspace, univariate only),
                    'lhs' (Latin hypercube).
            rng: Optional RandomState for deterministic sampling.
                 If None, uses the global numpy RNG.

        Returns:
            (n_points, n_vars + 1) array — input columns + output column.
        """
        n_vars = len(self._variables)
        _rng = rng if rng is not None else np.random

        if method == 'uniform':
            inputs = np.column_stack([
                _rng.uniform(v.low, v.high, n_points)
                for v in self._variables
            ])
        elif method == 'grid':
            if n_vars != 1:
                raise ValueError("'grid' sampling only works for univariate expressions")
            inputs = np.linspace(
                self._variables[0].low,
                self._variables[0].high,
                n_points,
            ).reshape(-1, 1)
        elif method == 'lhs':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n_vars)
            unit_samples = sampler.random(n=n_points)
            lows = np.array([v.low for v in self._variables])
            highs = np.array([v.high for v in self._variables])
            inputs = qmc.scale(unit_samples, lows, highs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        outputs = self.evaluate(inputs)
        return np.column_stack([inputs, outputs])

    def is_equivalent(self, other: 'Expression',
                      method: str = 'numeric',
                      n_test: int = 100,
                      rtol: float = 1e-6) -> bool:
        """Check equivalence with another expression.

        Args:
            method: 'numeric' (sample & compare), 'symbolic' (sympy simplify),
                    or 'both'.
        """
        if method == 'symbolic' or method == 'both':
            try:
                diff = sp.simplify(self._expr - other._expr)
                if diff.is_zero is True:
                    return True
                if method == 'symbolic':
                    return False
            except Exception:
                if method == 'symbolic':
                    return False

        # Numeric comparison
        try:
            cloud = self.sample(n_test, method='uniform')
            inputs = cloud[:, :-1]
            y_self = cloud[:, -1]
            y_other = other.evaluate(inputs)

            finite = np.isfinite(y_self) & np.isfinite(y_other)
            if finite.sum() < 10:
                return False
            return np.allclose(y_self[finite], y_other[finite], rtol=rtol, atol=1e-10)
        except Exception:
            return False

    def __repr__(self):
        return f"Expression({self._expr}, vars={[v.name for v in self._variables]})"


# ============================================================================
# Feynman loader
# ============================================================================

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    implicit_application,
)

_SYMBOLS_REQUIRING_SYM = [
    r'Volt', r'mob', r'mom', r'Bx', r'By', r'Bz', r'Nn',
    r'Int_0', r'k_spring', r'mu_drift', r'rho_c_0', r'sigma_den', r'A_vec',
    r'omega_0', r'p_d', r'n_0', r'n_rho', r'm_rho', r'g_', r'kb', r'Ef', r'Pwr',
    r'm_0', r'q1', r'q2', r'I', r'I1', r'I2', r'pr',
]

_SUBSCRIPTED_SYMS = [
    r'V([0-9])', r'T([0-9])', r'r([0-9])', r'x([0-9])', r'y([0-9])',
    r'z([0-9])', r'm([0-9])', r'd([0-9])', r'theta([0-9])',
]


def _clean_formula_string(formula: str) -> str:
    """Apply _sym suffixes to prevent SymPy misinterpretation."""
    for sym_name in _SYMBOLS_REQUIRING_SYM:
        formula = re.sub(r'\b' + sym_name + r'\b', sym_name + '_sym', formula)
    for sym_pattern in _SUBSCRIPTED_SYMS:
        formula = re.sub(r'\b(' + sym_pattern + r')\b', r'\1_sym', formula)
    formula = re.sub(r'\bpi\b', 'pi_sym', formula)
    formula = re.sub(r'\be\b', 'e_sym', formula)
    return formula


def _build_feynman_namespace():
    ns = {}

    sym_pairs = {
        'Volt_sym': 'Volt', 'mob_sym': 'mob', 'mom_sym': 'mom',
        'Bx_sym': 'Bx', 'By_sym': 'By', 'Bz_sym': 'Bz', 'Nn_sym': 'Nn',
        'Int_0_sym': 'Int_0', 'k_spring_sym': 'k_spring',
        'mu_drift_sym': 'mu_drift', 'rho_c_0_sym': 'rho_c_0',
        'sigma_den_sym': 'sigma_den', 'A_vec_sym': 'A_vec',
        'omega_0_sym': 'omega_0', 'p_d_sym': 'p_d',
        'n_0_sym': 'n_0', 'n_rho_sym': 'n_rho', 'm_rho_sym': 'm_rho',
        'g__sym': 'g_', 'kb_sym': 'kb', 'Ef_sym': 'Ef', 'Pwr_sym': 'Pwr',
        'm_0_sym': 'm_0', 'q1_sym': 'q1', 'q2_sym': 'q2',
        'I_sym': 'I', 'I1_sym': 'I1', 'I2_sym': 'I2', 'pr_sym': 'pr',
        'pi_sym': sp.pi, 'e_sym': sp.E,
    }
    for key, val in sym_pairs.items():
        ns[key] = val if isinstance(val, sp.Basic) else sp.Symbol(val)

    for prefix in ['V', 'T', 'r', 'x', 'y', 'z', 'm', 'd', 'theta']:
        for digit in range(10):
            name = f'{prefix}{digit}'
            ns[f'{name}_sym'] = sp.Symbol(name)

    for name in ['q', 'B', 'p', 'omega', 'theta', 'F', 'alpha', 'kappa',
                 'epsilon', 'chi', 'U', 'm', 'v', 'u', 'w', 'sigma', 'H',
                 'M', 'Y', 'A', 'n', 'd', 'C', 't', 'r', 'c', 'h', 'T',
                 'gamma', 'beta', 'delta', 'mu', 'E_n', 'Jz', 'rho', 'a',
                 'k', 'V', 'g', 'x', 'y', 'z']:
        ns[name] = sp.Symbol(name)

    ns['pi'] = sp.pi
    ns['e'] = sp.E

    return ns


_FEYNMAN_NAMESPACE = _build_feynman_namespace()


def load_feynman_csv(csv_path: str) -> list[Expression]:
    """Load Feynman equations from CSV into a list of Expression objects.

    Skips equations that fail to parse, with a warning.
    """
    df = pd.read_csv(csv_path)
    expressions = []

    for idx, row in df.iterrows():
        formula = row.get('Formula')
        if pd.isna(formula):
            continue
        try:
            expr = Expression.from_feynman_row(row)
            expressions.append(expr)
        except Exception as e:
            warnings.warn(f"Row {idx}: skipping '{formula}' — {e}")

    return expressions


# ============================================================================
# Synthetic expression loader
# ============================================================================

_SYNTHETIC_SYMPY_NS = {
    'sin': sp.sin, 'cos': sp.cos,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'exp': sp.exp, 'log': sp.log,
    'sqrt': sp.sqrt, 'abs': sp.Abs,
    'pi': sp.pi,
    'x': sp.Symbol('x'),
}


def _synthetic_string_to_sympy(expr_str: str) -> Optional[sp.Expr]:
    """Convert a SYMBA_Reg_Data_Gen expression string to SymPy.

    Expands sinc(...) and lorentz(...) into elementary ops before parsing.
    """
    s = expr_str

    def _expand_sinc(m):
        arg = m.group(1)
        return f"(sin({arg})/({arg}))"
    s = re.sub(r'sinc\(([^)]+)\)', _expand_sinc, s)

    def _expand_lorentz(m):
        arg = m.group(1)
        s_val = m.group(2)
        return f"(1/(1+({arg}/{s_val})**2))"
    s = re.sub(r'lorentz\(([^,]+),\s*([^)]+)\)', _expand_lorentz, s)

    try:
        return sp.sympify(s, locals=_SYNTHETIC_SYMPY_NS)
    except Exception:
        return None


def load_synthetic_pkl(
    pkl_path: str,
    max_seq_len: int = 64,
    tokenizer: Optional[PrefixTokenizer] = None,
    max_expressions: int = 0,
    dedupe_by_tokens: bool = True,
    max_per_sequence: int = 1,
    progress: bool = False,
) -> list[Expression]:
    """Load synthetic expressions from a pickle file.

    The pickle should contain a list of expression strings as produced
    by the SYMBA_Reg_Data_Gen notebook.

    Args:
        pkl_path: Path to expressions.pkl.
        max_seq_len: Drop expressions whose prefix tokenization exceeds this.
        tokenizer: PrefixTokenizer instance (created if None).
        max_expressions: Max number of expressions to load (0 = all).
        dedupe_by_tokens: Keep at most *max_per_sequence* expressions per
            distinct prefix token sequence.
        max_per_sequence: Copies to keep per distinct token sequence.
        progress: Show a tqdm bar over the raw strings.  Each one is parsed
            with SymPy and converted to prefix notation, so this loop is the
            slow part of loading a large pickle.

    Returns:
        List of Expression objects ready for dataset construction.
    """
    if tokenizer is None:
        tokenizer = PrefixTokenizer()

    with open(pkl_path, 'rb') as f:
        expr_strings: list[str] = pickle.load(f)

    if max_expressions > 0:
        expr_strings = expr_strings[:max_expressions]

    var_meta = [VarMeta(name='x', low=-math.pi, high=math.pi)]

    results: list[Expression] = []
    n_parse = 0
    n_tokenize = 0
    n_too_long = 0
    n_unk = 0
    n_dup = 0
    seq_counts: dict[tuple, int] = {}
    first_errors: list[str] = []

    src = expr_strings
    if progress:
        from tqdm.auto import tqdm
        src = tqdm(expr_strings, desc='parsing expressions', leave=False)

    for expr_str in src:
        sympy_expr = _synthetic_string_to_sympy(expr_str)
        if sympy_expr is None:
            n_parse += 1
            if len(first_errors) < 3:
                first_errors.append(f"parse: {expr_str[:80]}")
            continue

        try:
            expr = Expression.from_sympy(sympy_expr, var_meta)
            ids = expr.tokenize(tokenizer)
        except (ValueError, Exception) as e:
            n_tokenize += 1
            if len(first_errors) < 3:
                first_errors.append(f"tokenize: {e}")
            continue

        if len(ids) > max_seq_len:
            n_too_long += 1
            continue
        if tokenizer.unk_id in ids:
            n_unk += 1
            if len(first_errors) < 3:
                unk_toks = [t for t in expr.prefix.split() if t not in tokenizer.token2id]
                first_errors.append(f"unk tokens: {unk_toks}")
            continue

        if dedupe_by_tokens:
            key = tuple(ids)
            if seq_counts.get(key, 0) >= max_per_sequence:
                n_dup += 1
                continue
            seq_counts[key] = seq_counts.get(key, 0) + 1

        results.append(expr)

    if dedupe_by_tokens:
        print(f"load_synthetic_pkl: {len(expr_strings)} strings -> "
              f"{len(results)} kept, {len(seq_counts)} distinct token sequences "
              f"({n_dup} duplicate-sequence expressions dropped)")

    n_failed = n_parse + n_tokenize + n_too_long + n_unk
    if n_failed > 0:
        detail = (f"parse={n_parse}, tokenize={n_tokenize}, "
                  f"too_long={n_too_long}, unk={n_unk}")
        examples = "; ".join(first_errors) if first_errors else ""
        warnings.warn(
            f"Skipped {n_failed}/{len(expr_strings)} expressions "
            f"({detail}). Examples: {examples}"
        )

    return results
