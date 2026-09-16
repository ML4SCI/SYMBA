"""Canonical expression templates with dynamic constant augmentation.

The synthetic pickle stores *instantiated* expressions: every fittable
coefficient is already a concrete float.  Deduplicating 200 000 of those by
prefix token sequence leaves only ~1 500 distinct canonical forms, so each
canonical form the decoder has to produce is backed by exactly **one**
numerical realisation.  The encoder can therefore memorise a coefficient
pattern instead of the structure that the target tokens actually describe.

This module separates the two:

.. code-block:: text

    canonical structure          c0*sin(c1*x1) + c2      (ConstantTemplate)
        |  sample constants
        v
    numerical function           2.1*sin(0.8*x1) + 4.3   (Expression)
        |  sample x points
        v
    point cloud                  -> encoder

    decoder target               C sin C x1 + C          (unchanged, always)

Two invariants hold by construction:

* the decoder target is derived from the **template**, never from a
  realisation, so coefficient values cannot perturb it;
* the split unit is the canonical form, so a structure seen in training can
  never reappear in val/test under different coefficients.

**Coefficient slots are independent, including across repeated subexpressions.**
The generator's ``sinc(k*x + phi)`` expands to ``sin(k*x+phi)/(k*x+phi)``, whose
four literals templatize to four *separate* slots, so a resampled realisation is
generally ``sin(a*x+b)/(c*x+d)`` rather than a sinc.  That is deliberate: the
decoder target already spells those positions as four independent ``C`` tokens,
and :func:`~symbolic_jepa.evaluation.fit_constants` fits them independently when
scoring.  Tying them would train the encoder on a correlation the target cannot
express and the evaluator does not assume.  The cost is that realisations range
over a wider family than the generator's — poles that sinc's removable
singularity avoided, for instance — which is what :func:`pool_is_usable` and the
rejection loop are there to catch.

Everything here is additive.  ``expressions.py``, ``tokenizer.py`` and
``dataset.py`` are untouched, so existing datasets, notebooks, checkpoints and
the content-addressed caches keyed on those modules' source all keep working
unchanged.  Dynamic constants are off by default at every entry point
(``dynamic_constants=False``); an experiment opts in explicitly.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sympy as sp
import torch

from symbolic_jepa.dataset import (
    _maybe_tqdm,
    _report_leakage,
    _split_indices,
    PointCloudDataset,
    sample_pool,
    subsample_and_normalize,
)
from symbolic_jepa.expressions import (
    Expression,
    VarMeta,
    _synthetic_string_to_sympy,
)
from symbolic_jepa.tokenizer import (
    SPECIFIC_NUMERICS,
    PrefixTokenizer,
    sympy_to_prefix,
)

__all__ = [
    'AUGMENTATION_VERSION',
    'augmentation_seed',
    'ConstantTemplate',
    'ConstantSampler',
    'DynamicConstantPointCloudDataset',
    'DynamicConstantMultiViewDataset',
    'InstantiatedExpression',
    'TEMPLATE_FORMAT',
    'audit_constant_sampling',
    'build_multiview_template_splits',
    'build_template_splits',
    'build_templates_from_strings',
    'canonical_split_report',
    'describe_realizations',
    'load_template_dataset',
    'pool_is_usable',
    'save_template_dataset',
    'stage2_fingerprint',
    'templates_fingerprint',
    'templatize',
]

TEMPLATE_FORMAT = 'symbolic_jepa.templates/v1'

# Bump when the mapping from (base_seed, stage, epoch, idx) to a realisation
# changes.  Stored in the checkpoint so a resume cannot silently continue under
# different augmentation than the epochs already trained.
AUGMENTATION_VERSION = 'stateless-v1'


def augmentation_seed(base_seed: int, stage, epoch: int, idx: int,
                      stream: str, attempt: int = 0) -> int:
    """Stable 32-bit seed for one (stage, epoch, example, stream, attempt).

    SHA-256 rather than Python's ``hash()``, which is randomised per process for
    strings and would make every fresh runtime a different experiment.

    *stream* separates concerns that must not perturb each other.  Constant
    sampling and point sampling draw from independent streams, so changing how
    many random values a coefficient draw consumes — which happens as soon as a
    form's slot count changes — cannot shift the point cloud.  *attempt* does
    the same across rejection retries: retry *k* is a fixed function of *k*, not
    of how much entropy attempts ``0..k-1`` happened to burn.
    """
    key = f'{base_seed}|{stage}|{epoch}|{idx}|{stream}|{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')


# ============================================================================
# Templatisation
# ============================================================================

def templatize(expr: sp.Expr) -> tuple[sp.Expr, list[float]]:
    """Split *expr* into a coefficient template and its constant values.

    Every numeric literal that :func:`sympy_to_prefix` would emit as the
    fittable ``C`` token is replaced by a fresh symbol ``c_0``, ``c_1``, ... in
    the order the prefix walk visits it.  Structural numerics — the exponents
    in :data:`SPECIFIC_NUMERICS`, ``pi`` and ``e`` — are left alone, exactly as
    the tokenizer treats them, so ``x**2`` stays ``x**2`` and never becomes a
    resampleable coefficient.

    The tree is rebuilt with ``evaluate=False``.  SymPy's automatic
    canonicalisation orders ``Add``/``Mul`` arguments by a sort key that
    depends on whether an argument is a number or a symbol, so an evaluated
    rebuild could permute arguments and change the prefix string.  Suppressing
    evaluation preserves the argument order of the input tree, which makes the
    template's prefix identical to the original expression's — verified over
    the full synthetic pickle by ``build_templates_from_strings``.

    Returns:
        ``(template_expr, values)`` where ``values[k]`` is the float that
        ``c_k`` replaced.
    """
    values: list[float] = []

    def _rec(node):
        if node == sp.pi or node == sp.E:
            return node
        if node.is_Number:
            if node in SPECIFIC_NUMERICS:
                return node
            sym = sp.Symbol(f'c_{len(values)}')
            values.append(float(node))
            return sym
        if node.is_Symbol:
            return node
        return node.func(*[_rec(a) for a in node.args], evaluate=False)

    return _rec(expr), values


class InstantiatedExpression(Expression):
    """One numerical realisation of a :class:`ConstantTemplate`.

    Behaves like an :class:`Expression` for sampling and evaluation, but its
    ``prefix`` comes from the template rather than from its own SymPy tree.
    That is the point: the decoder target must be a function of the structure
    alone.  ``sympy_expr`` is materialised lazily (only diagnostics need it)
    because substituting into a SymPy tree costs far more than evaluating one.
    """

    def __init__(self, template: 'ConstantTemplate', values: Sequence[float]):
        super().__init__(None, template.variables)
        self.template = template
        self.values = np.asarray(values, dtype=float)
        if len(self.values) != template.n_constants:
            raise ValueError(
                f'{template.n_constants} constant slots but '
                f'{len(self.values)} values supplied'
            )
        self._prefix = template.prefix

    @property
    def sympy_expr(self) -> sp.Expr:
        if self._expr is None:
            self._expr = self.template.substitute(self.values)
        return self._expr

    @property
    def prefix(self) -> str:
        return self.template.prefix

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return self.template.evaluate(points, self.values)

    def __repr__(self):
        return (f'InstantiatedExpression({self.template.prefix} | '
                f'{np.array2string(self.values, precision=3)})')


class ConstantTemplate:
    """A canonical symbolic structure with resampleable coefficient slots.

    Args:
        template_expr: SymPy tree whose fittable constants are ``c_0..c_{k-1}``.
        reference_expr: The original SymPy tree with its generator-drawn
            floats still in place.  Kept verbatim rather than reconstructed by
            substitution so that the reference realisation is bit-identical to
            what the pre-template pipeline used — that is what makes
            ``dynamic_constants=False`` provably a no-op.
        reference_constants: The floats ``c_0..c_{k-1}`` stand for in
            *reference_expr*, slot-aligned.
        variables: Input variables and their sampling bounds.
        constant_pools: Optional per-slot arrays of values observed for this
            canonical form across the whole raw corpus.  ``constant_pools[k]``
            feeds slot ``k``.
        prefix: Precomputed canonical prefix string, if already known.
    """

    def __init__(
        self,
        template_expr: sp.Expr,
        reference_expr: sp.Expr,
        reference_constants: Sequence[float],
        variables: Sequence[VarMeta],
        constant_pools: Optional[Sequence[np.ndarray]] = None,
        prefix: Optional[str] = None,
    ):
        self.expr = template_expr
        self.reference_expr = reference_expr
        self.reference_constants = np.asarray(reference_constants, dtype=float)
        self.variables = list(variables)
        self.constant_pools = (
            None if constant_pools is None
            else [np.asarray(p, dtype=np.float32) for p in constant_pools]
        )
        self._prefix = prefix
        self._fn = None  # lambdify cache over (vars..., consts...)

    # --- Pickling: the lambdify cache is derived and not picklable ---

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fn'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fn = None

    # --- Constructors ---

    @classmethod
    def from_sympy(cls, expr: sp.Expr, variables: Sequence[VarMeta],
                   constant_pools=None) -> 'ConstantTemplate':
        template_expr, values = templatize(expr)
        return cls(template_expr, expr, values, variables,
                   constant_pools=constant_pools)

    @classmethod
    def from_expression(cls, expression: Expression) -> 'ConstantTemplate':
        return cls.from_sympy(expression.sympy_expr, expression.variables)

    # --- Structure ---

    @property
    def n_constants(self) -> int:
        return len(self.reference_constants)

    @property
    def const_symbols(self) -> list[sp.Symbol]:
        return [sp.Symbol(f'c_{i}') for i in range(self.n_constants)]

    @property
    def var_map(self) -> dict[str, str]:
        return {v.name: f'x{i + 1}' for i, v in enumerate(self.variables)}

    @property
    def prefix(self) -> str:
        """Canonical prefix string — every coefficient slot renders as ``C``."""
        if self._prefix is None:
            var_map = dict(self.var_map)
            for i in range(self.n_constants):
                var_map[f'c_{i}'] = 'C'
            p = sympy_to_prefix(self.expr, var_map)
            if p is None:
                raise ValueError(f'Cannot convert template to prefix: {self.expr}')
            self._prefix = p
        return self._prefix

    def token_ids(self, tokenizer: PrefixTokenizer) -> list[int]:
        return tokenizer.encode(self.prefix)

    # --- Realisation ---

    def _callable(self):
        if self._fn is None:
            args = [sp.Symbol(v.name) for v in self.variables] + self.const_symbols
            self._fn = sp.lambdify(args, self.expr, 'numpy')
        return self._fn

    def evaluate(self, points: np.ndarray, values: Sequence[float]) -> np.ndarray:
        """Evaluate the template at *points* with coefficients *values*.

        The template is lambdified **once**, with the coefficients as extra
        arguments, so a fresh realisation costs one array call rather than a
        SymPy substitution plus a fresh ``lambdify`` (a few ms of codegen that
        would otherwise be paid on every ``__getitem__``).
        """
        fn = self._callable()
        result = fn(*points.T, *values)
        return np.broadcast_to(
            np.asarray(result, dtype=float), (points.shape[0],)).copy()

    def substitute(self, values: Sequence[float]) -> sp.Expr:
        """The SymPy tree with *values* substituted in.  For display only."""
        return self.expr.subs(
            dict(zip(self.const_symbols, [sp.Float(v) for v in values])))

    def instantiate(self, values: Sequence[float]) -> InstantiatedExpression:
        """A fast :class:`Expression` for this template at *values*."""
        return InstantiatedExpression(self, values)

    def reference_expression(self) -> Expression:
        """The original generator realisation, as a plain :class:`Expression`.

        Returns the untouched pre-template object, so point clouds built from
        it match the pre-template pipeline exactly.
        """
        return Expression.from_sympy(self.reference_expr, self.variables)

    def __repr__(self):
        return (f'ConstantTemplate({self.prefix} | {self.n_constants} slots'
                f'{"" if self.constant_pools is None else ", pooled"})')


# ============================================================================
# Constant sampling
# ============================================================================

class ConstantSampler:
    """Draws coefficient vectors for a :class:`ConstantTemplate`.

    **Mode ``'global'`` (the default) is the one the experiment needs.** Every
    coefficient of every canonical form draws its magnitude from a *single*
    shared pool, so no form's coefficients carry information about which form it
    is.  That matters more than distributional fidelity: if form *F* drew from
    *F*'s own observed values, *F*'s coefficients would stay a fingerprint for
    *F*, which is precisely the shortcut this augmentation exists to destroy.
    The pool is fitted on the **training** templates only (see :meth:`fit`).

    The **sign** comes from the form's reference coefficient rather than from
    the pool.  Sign is structural, not a nuisance value: ``exp(c*(x-mu)**2)``
    is a Gaussian bump when ``c < 0`` and a divergence when ``c > 0``, so a free
    sign makes a large fraction of draws unusable.  Measured over 400 training
    forms at 4 draws each, on the 200k-string corpus:

    ======================================  ============  =================
    sampler                                 accept rate   forms starved
    ======================================  ============  =================
    global signed pool, free sign                  0.749             4.2 %
    global magnitude pool + reference sign         0.980             0.0 %
    per-form pool (``'empirical'``)                0.997             0.2 %
    ======================================  ============  =================

    Rejection under a free sign simply re-imposes the sign constraint after the
    fact, and does so unevenly across forms — so it buys no extra uniformity,
    only a starved tail.

    ``'empirical'`` keeps the old per-form behaviour (each slot drawn from the
    values observed for *that* slot of *that* form).  Retained for ablation: it
    is the form-biased comparison point, not a default.

    ``'perturb'`` needs no pools at all — it scales each reference coefficient
    by a log-uniform factor in ``[1/log_spread, log_spread]``, sign preserved.
    It is also the per-slot fallback under ``'empirical'`` when a pool is too
    thin to sample from.

    Args:
        mode: ``'global'``, ``'empirical'`` or ``'perturb'``.
        log_spread: Magnitude spread for ``'perturb'`` (and for *jitter*).
        jitter: Log-uniform factor applied on top of a pooled draw.  ``1.0``
            (default) leaves draws exactly as observed; a value like ``1.1``
            makes the augmentation continuous rather than a finite grid.
        magnitude_pool: Pre-fitted global magnitude pool.  Usually left ``None``
            and filled in by :meth:`fit`.
    """

    def __init__(self, mode: str = 'global', log_spread: float = 2.0,
                 jitter: float = 1.0, magnitude_pool=None):
        if mode not in ('global', 'empirical', 'perturb'):
            raise ValueError(f'Unknown constant sampling mode: {mode!r}')
        if log_spread <= 1.0:
            raise ValueError('log_spread must be > 1')
        if jitter < 1.0:
            raise ValueError('jitter must be >= 1')
        self.mode = mode
        self.log_spread = float(log_spread)
        self.jitter = float(jitter)
        self.magnitude_pool = (
            None if magnitude_pool is None
            else np.asarray(magnitude_pool, dtype=np.float32))

    # --- Fitting the shared pool ---

    @property
    def is_fitted(self) -> bool:
        return self.mode != 'global' or self.magnitude_pool is not None

    def fit(self, templates: Sequence[ConstantTemplate]) -> 'ConstantSampler':
        """Build the shared magnitude pool from *templates*.

        Pass the **training** templates only.  The pool is a distribution over
        coefficient magnitudes rather than over structures, so including held-out
        forms would not leak a target — but it costs nothing to exclude them and
        removes the question entirely.

        Every observed value of every slot of every form contributes, so the
        result is the corpus's own magnitude distribution and no range has to be
        chosen by hand.  Zero and non-finite magnitudes are dropped: a zero
        coefficient erases a branch, which is one of the pathologies
        :func:`pool_is_usable` exists to reject.
        """
        parts = []
        for t in templates:
            if t.constant_pools:
                parts.extend(np.abs(np.asarray(p, dtype=np.float64))
                             for p in t.constant_pools)
            elif t.n_constants:
                parts.append(np.abs(t.reference_constants))

        if not parts:
            raise ValueError('no coefficient values to fit a global pool from')

        pool = np.concatenate(parts)
        pool = pool[np.isfinite(pool) & (pool > 0)]
        if pool.size == 0:
            raise ValueError('every observed coefficient magnitude was zero '
                             'or non-finite')
        self.magnitude_pool = pool.astype(np.float32)
        return self

    def describe(self) -> str:
        if self.mode != 'global':
            return f'{self.mode} (no shared pool)'
        if self.magnitude_pool is None:
            return 'global (unfitted)'
        p = self.magnitude_pool
        q = np.percentile(p, [50, 90, 99])
        return (f'global magnitude pool: {p.size} values, '
                f'median {q[0]:.3f}, p90 {q[1]:.3f}, p99 {q[2]:.3f}, '
                f'max {p.max():.1f}')

    # --- Drawing ---

    def _jitter(self, values: np.ndarray, rng) -> np.ndarray:
        if self.jitter <= 1.0:
            return values
        log_j = np.log(self.jitter)
        return values * np.exp(rng.uniform(-log_j, log_j, values.shape))

    def _signs(self, template: ConstantTemplate) -> np.ndarray:
        signs = np.sign(template.reference_constants)
        signs[signs == 0] = 1.0
        return signs

    def sample(self, template: ConstantTemplate, rng=None) -> np.ndarray:
        """One coefficient vector, slot-aligned with ``template``."""
        src = rng if rng is not None else np.random
        n = template.n_constants
        if n == 0:
            return np.empty(0, dtype=float)

        if self.mode == 'global':
            if self.magnitude_pool is None:
                raise RuntimeError(
                    "ConstantSampler(mode='global') must be fitted before use; "
                    "call sampler.fit(train_templates) or let "
                    "build_template_splits do it."
                )
            mags = self.magnitude_pool[
                src.randint(self.magnitude_pool.size, size=n)
            ].astype(float)
            return self._jitter(mags, src) * self._signs(template)

        log_s = np.log(self.log_spread)
        if self.mode == 'perturb':
            return template.reference_constants * np.exp(
                src.uniform(-log_s, log_s, n))

        # 'empirical': per-form, per-slot pools, with a perturb fallback for
        # any slot too thin to sample from.
        pools = template.constant_pools
        out = np.empty(n, dtype=float)
        for k in range(n):
            pool = None
            if pools is not None and k < len(pools) and pools[k].size >= 2:
                pool = pools[k]
            if pool is None:
                out[k] = float(template.reference_constants[k]) * float(
                    np.exp(src.uniform(-log_s, log_s)))
            else:
                out[k] = float(pool[src.randint(pool.size)])
        return self._jitter(out, src)


def pool_is_usable(pool: np.ndarray, max_abs: float = 1e6,
                   min_std: float = 1e-8, min_rel_std: float = 1e-9) -> bool:
    """Whether a finite-filtered point cloud is a usable training example.

    ``sample_pool`` has already dropped non-finite rows and guaranteed a minimum
    count, which is the existing validity rule.  Two failure modes specific to
    resampled coefficients survive it:

    * a coefficient near zero **erases a branch**, leaving an output column that
      is constant.  ``subsample_and_normalize`` divides by ``std + 1e-8``, so a
      degenerate column becomes numerical noise amplified by 1e8;
    * a coefficient blows the output up to a magnitude that is finite but far
      outside the generator's plausible range.

    The magnitude bound mirrors the amplitude guard in the generator's own
    ``is_plausible`` filter.  Degeneracy is checked both absolutely and relative
    to the output scale, so a genuinely small-amplitude function is kept while a
    flat one is not.
    """
    y = pool[:, -1]
    if not np.isfinite(y).all():
        return False
    peak = float(np.max(np.abs(y)))
    if peak > max_abs:
        return False
    std = float(np.std(y))
    return std > min_std and std >= min_rel_std * peak


# ============================================================================
# Building templates from the raw synthetic corpus
# ============================================================================

def build_templates_from_strings(
    expr_strings: Sequence[str],
    tokenizer: PrefixTokenizer,
    *,
    max_seq_len: int = 64,
    variables: Optional[Sequence[VarMeta]] = None,
    pool_cap: int = 512,
    form_horizon: int = 0,
    seed: int = 42,
    progress: bool = False,
) -> tuple[list[ConstantTemplate], dict]:
    """Deduplicate raw expression strings into canonical templates.

    Mirrors :func:`load_synthetic_pkl` with ``dedupe_by_tokens=True,
    max_per_sequence=1`` filter-for-filter and in the same order, so the
    resulting template list is positionally aligned with the deduplicated
    ``Expression`` list the existing pipeline builds.  Given the same ``seed``,
    ``_split_indices`` therefore produces the *same* train/val/test partition,
    and a dynamic-constants run stays comparable to the run it is measured
    against.

    Unlike that loader, the strings that dedupe away are not discarded: each
    one contributes its coefficients to the surviving template's per-slot
    pools.  The k-th pool entry is well defined because the k-th ``C`` of an
    identical prefix string is the same tree position.  200 000 raw strings
    collapse to ~1 500 forms, so the average form arrives with ~130 observed
    realisations per slot instead of one.

    Args:
        expr_strings: Raw generator strings.
        tokenizer: Tokenizer whose vocabulary defines the canonical form.
        max_seq_len: Drop forms whose tokenization exceeds this.
        variables: Variable metadata; defaults to the univariate ``x`` on
            ``[-pi, pi]`` that the synthetic generator uses.
        pool_cap: Cap on values retained per slot.  Pools are reservoir-free:
            the first ``pool_cap`` observations are kept, which is unbiased
            because the corpus order is already the generator's random order.
        form_horizon: If > 0, only the first ``form_horizon`` strings may
            introduce a *new* canonical form; later ones may still feed the
            pools of forms already found.  This decouples the two things corpus
            size controls.  Reading more strings normally enlarges the form set,
            which changes the train/val/test split and breaks comparability with
            the baseline run; setting ``form_horizon`` to the baseline's
            ``MAX_SYNTH`` freezes the split while a longer read deepens the
            coefficient pools.
        seed: Reserved for reproducibility of future subsampling policies.
        progress: Show a tqdm bar (this loop is SymPy-bound, ~3 ms/string).

    Returns:
        ``(templates, stats)``.
    """
    if variables is None:
        variables = [VarMeta(name='x', low=-math.pi, high=math.pi)]
    variables = list(variables)
    var_map = {v.name: f'x{i + 1}' for i, v in enumerate(variables)}

    templates: list[ConstantTemplate] = []
    pools: list[list[list[float]]] = []   # templates[i] -> slot -> values
    index: dict[tuple, int] = {}

    stats = {
        'n_strings': len(expr_strings),
        'n_parse_failed': 0,
        'n_tokenize_failed': 0,
        'n_too_long': 0,
        'n_unk': 0,
        'n_prefix_mismatch': 0,
        'n_slot_mismatch': 0,
        'n_pooled': 0,
        'n_beyond_horizon': 0,
        'form_horizon': int(form_horizon),
    }

    src = _maybe_tqdm(expr_strings, progress, 'building canonical templates')
    for position, expr_str in enumerate(src):
        sympy_expr = _synthetic_string_to_sympy(expr_str)
        if sympy_expr is None:
            stats['n_parse_failed'] += 1
            continue

        # The instantiated prefix is the existing dedup key; computing it the
        # same way keeps this loop's accept/reject decisions identical to
        # load_synthetic_pkl's.
        instantiated_prefix = sympy_to_prefix(sympy_expr, var_map)
        if instantiated_prefix is None:
            stats['n_tokenize_failed'] += 1
            continue
        ids = tokenizer.encode(instantiated_prefix)
        if len(ids) > max_seq_len:
            stats['n_too_long'] += 1
            continue
        if tokenizer.unk_id in ids:
            stats['n_unk'] += 1
            continue

        template_expr, values = templatize(sympy_expr)
        key = tuple(ids)
        slot = index.get(key)

        if slot is None:
            if form_horizon and position >= form_horizon:
                stats['n_beyond_horizon'] += 1
                continue
            tmpl = ConstantTemplate(
                template_expr, sympy_expr, values, variables)
            # The template must render to the same tokens as the realisation it
            # came from; otherwise the decoder target would silently shift for
            # this form.  Drop the form rather than train on a mismatch.
            if tmpl.prefix != instantiated_prefix:
                stats['n_prefix_mismatch'] += 1
                continue
            index[key] = len(templates)
            templates.append(tmpl)
            pools.append([[v] for v in values])
            continue

        # A duplicate of an already-seen canonical form: harvest its constants.
        target = pools[slot]
        if len(values) != len(target):
            stats['n_slot_mismatch'] += 1
            continue
        stats['n_pooled'] += 1
        for k, v in enumerate(values):
            if len(target[k]) < pool_cap:
                target[k].append(v)

    for tmpl, slots in zip(templates, pools):
        tmpl.constant_pools = [np.asarray(s, dtype=np.float32) for s in slots]

    sizes = [len(s[0]) for s in pools if s]
    stats['n_templates'] = len(templates)
    stats['pool_min'] = int(min(sizes)) if sizes else 0
    stats['pool_median'] = float(np.median(sizes)) if sizes else 0.0
    stats['pool_max'] = int(max(sizes)) if sizes else 0
    stats['n_single_realization'] = sum(1 for s in sizes if s < 2)

    n_failed = (stats['n_parse_failed'] + stats['n_tokenize_failed']
                + stats['n_too_long'] + stats['n_unk'])
    if n_failed:
        warnings.warn(
            f"Skipped {n_failed}/{len(expr_strings)} expressions "
            f"(parse={stats['n_parse_failed']}, "
            f"tokenize={stats['n_tokenize_failed']}, "
            f"too_long={stats['n_too_long']}, unk={stats['n_unk']})."
        )
    if stats['n_prefix_mismatch'] or stats['n_slot_mismatch']:
        warnings.warn(
            f"Dropped {stats['n_prefix_mismatch']} form(s) whose template "
            f"prefix differed from the realisation's, and "
            f"{stats['n_slot_mismatch']} realisation(s) with a mismatched "
            f"slot count."
        )
    return templates, stats


# ============================================================================
# On-disk template dataset
# ============================================================================

def save_template_dataset(path, templates: Sequence[ConstantTemplate],
                          tokenizer: PrefixTokenizer, meta: Optional[dict] = None):
    """Write a template dataset, atomically.

    Deduplication is the expensive step (SymPy, ~3 ms per raw string, ~13 min
    for 200 000) and its result is a property of the corpus, not of any run.
    It is done once here and every experiment loads the canonical forms
    directly.  Writes a new file: the source pickle is never modified.
    """
    path = Path(path)
    payload = {
        'format': TEMPLATE_FORMAT,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'tokenizer_vocab': tuple(tokenizer.vocab),
        'templates': list(templates),
        'meta': dict(meta or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def templates_fingerprint(templates: Sequence[ConstantTemplate]) -> str:
    """Short stable hash of a template list's identity and order.

    Stored in each checkpoint so a resume cannot silently continue against a
    different canonical-template dataset — a rebuilt file with a different
    ``--max-expressions`` would change both the forms and the split, and the
    weights would load without complaint.

    Covers the canonical prefixes (which fix the decoder targets), their order
    (which fixes the split), and each form's reference coefficients (which fix
    the val/test point clouds).
    """
    h = hashlib.sha256()
    for t in templates:
        h.update(t.prefix.encode())
        h.update(b'\0')
        h.update(np.asarray(t.reference_constants, dtype=np.float64).tobytes())
        h.update(b'\n')
    h.update(str(len(templates)).encode())
    return h.hexdigest()[:16]


def load_template_dataset(path, tokenizer: Optional[PrefixTokenizer] = None,
                          max_templates: int = 0
                          ) -> tuple[list[ConstantTemplate], dict]:
    """Read a template dataset written by :func:`save_template_dataset`.

    Raises on a format mismatch, and on a tokenizer whose vocabulary differs
    from the one the file was built with — the canonical forms are defined by
    that vocabulary, so a silent mismatch would mean training on targets the
    tokenizer cannot reproduce.
    """
    with open(path, 'rb') as f:
        payload = pickle.load(f)

    fmt = payload.get('format')
    if fmt != TEMPLATE_FORMAT:
        raise ValueError(
            f'{path}: expected format {TEMPLATE_FORMAT!r}, found {fmt!r}')

    if tokenizer is not None:
        stored = tuple(payload.get('tokenizer_vocab', ()))
        if stored != tuple(tokenizer.vocab):
            raise ValueError(
                f'{path} was built with a different tokenizer vocabulary '
                f'({len(stored)} tokens vs {len(tokenizer.vocab)}); rebuild it '
                f'with tools/build_template_dataset.py'
            )

    templates = payload['templates']
    if max_templates > 0:
        templates = templates[:max_templates]

    meta = dict(payload.get('meta', {}))
    meta['created'] = payload.get('created')
    meta['n_templates_in_file'] = len(payload['templates'])
    return list(templates), meta


# ============================================================================
# Dataset
# ============================================================================

class DynamicConstantPointCloudDataset(PointCloudDataset):
    """Point clouds from canonical templates, with freshly sampled constants.

    Every ``__getitem__`` draws a new coefficient vector, instantiates the
    function, and samples a fresh cloud from it.  The token target is untouched:
    it was built once from the template's canonical prefix.  Dataset length does
    not change, so epochs and optimizer steps stay comparable to the baseline.

    ``dynamic_constants`` is **forced off wherever ``resample`` is off**, so a
    val/test dataset is always fixed structures, fixed coefficients, fixed
    clouds — a moving evaluation set could not be compared across epochs or
    against the baseline.  Turning it off on a training set is still available
    as a same-notebook control: the dataset then serves each template's
    *reference* realisation, i.e. the exact ``Expression`` objects the
    pre-template pipeline used, reproducing the old behaviour rather than
    approximating it.  (Backward compatibility does not depend on that:
    ``build_synthetic_splits`` and ``data/synthetic.pkl`` are untouched.)

    Args:
        templates: Canonical templates.
        tokenizer: Tokenizer for the decoder targets.
        dynamic_constants: Resample coefficients per draw.  Silently ignored
            when ``resample=False``.
        sampler: :class:`ConstantSampler`; a default ``'global'`` one is built
            and fitted on *templates* if omitted.  Pass a shared instance to
            control the fit yourself.
        constant_seed: ``None`` (default) draws from the global NumPy RNG,
            matching how the existing point-cloud augmentation is seeded (per
            worker, via ``seed_worker``).  An int instead derives coefficients
            deterministically from ``(constant_seed, epoch, idx)``; set
            ``dataset.epoch`` each epoch and build the loader with
            ``persistent_workers=False``, or forked workers will never see the
            update.
        max_constant_tries: Redraws allowed before falling back to the
            reference coefficients, which are known-good.
        pool_mult: Candidate points drawn before subsampling (1 matches the
            single-view training path).
        **kwargs: Forwarded to :class:`PointCloudDataset`.
    """

    def __init__(
        self,
        templates: Sequence[ConstantTemplate],
        tokenizer: PrefixTokenizer,
        *,
        dynamic_constants: bool = True,
        sampler: Optional[ConstantSampler] = None,
        constant_seed: Optional[int] = None,
        max_constant_tries: int = 8,
        pool_mult: int = 1,
        max_abs: float = 1e6,
        **kwargs,
    ):
        by_id = {}
        reference_exprs = []
        for tmpl in templates:
            expr = tmpl.reference_expression()
            by_id[id(expr)] = tmpl
            reference_exprs.append(expr)

        super().__init__(reference_exprs, tokenizer, **kwargs)

        # PointCloudDataset drops expressions that fail tokenization or the
        # point-cloud probe, so recover the template of each survivor.
        for s in self.samples:
            s['template'] = by_id[id(s['expr'])]

        self.dynamic_constants = bool(dynamic_constants) and self.resample
        self.sampler = sampler or ConstantSampler()
        self.constant_seed = constant_seed

        # A 'global' sampler needs its shared pool before the first draw. Fit it
        # on THIS dataset's templates, which for the training split is exactly
        # the training templates — a val/test dataset never reaches this branch
        # because dynamic_constants is forced off wherever resample is off.
        if self.dynamic_constants and not self.sampler.is_fitted:
            self.sampler.fit([s['template'] for s in self.samples])

        self.max_constant_tries = int(max_constant_tries)
        self.pool_mult = int(pool_mult)
        self.max_abs = float(max_abs)
        self.epoch = 0
        self.stage = None

        # Per-process counters.  With num_workers > 0 each worker keeps its own
        # copy and the parent's stay at zero — use audit_constant_sampling for
        # a figure that is actually observable from the notebook.
        self.n_constant_draws = 0
        self.n_constant_rejections = 0
        self.n_constant_fallbacks = 0

    @property
    def templates(self) -> list[ConstantTemplate]:
        return [s['template'] for s in self.samples]

    @property
    def canonical_forms(self) -> list[str]:
        """Canonical prefix string of every sample, in dataset order."""
        return [s['template'].prefix for s in self.samples]

    def unique_canonical_count(self) -> int:
        return len(set(self.canonical_forms))

    # --- Deterministic augmentation state ---

    @property
    def deterministic_augmentation(self) -> bool:
        """Whether realisations are a pure function of (seed, stage, epoch, idx).

        True exactly when ``constant_seed`` is set.  With it unset the dataset
        draws from the global NumPy RNG, which is worker- and history-dependent
        and therefore not reproducible across a resume.
        """
        return self.constant_seed is not None

    def set_epoch(self, epoch: int, stage=None) -> 'DynamicConstantPointCloudDataset':
        """Point the augmentation at *epoch* (and optionally a training stage).

        Call before each epoch.  Under deterministic augmentation this is what
        advances the realisations — without it every epoch replays the same
        coefficients.  *stage* keeps Stage 1 and Stage 2 from handing the same
        example identical coefficients at the same epoch number.
        """
        self.epoch = int(epoch)
        if stage is not None:
            self.stage = stage
        return self

    def _stream_rngs(self, idx: int, attempt: int, fallback):
        """``(constants_rng, points_rng, subsample_rng)`` for one attempt.

        Returns *fallback* three times when ``constant_seed`` is None, which
        preserves the original single-stream stochastic behaviour exactly.
        """
        if self.constant_seed is None:
            return fallback, fallback, fallback
        return tuple(
            np.random.RandomState(augmentation_seed(
                self.constant_seed, self.stage, self.epoch, idx,
                stream, attempt))
            for stream in ('constants', 'points', 'subsample')
        )

    def draw_realization(self, idx: int, rng=None):
        """Sample coefficients for *idx* and return a validated point pool.

        Returns ``(pool, expr, n_rejected, used_fallback, subsample_rng)``.
        """
        s = self.samples[idx]
        tmpl = s['template']
        n_points = self.n_points * self.pool_mult
        rejected = 0
        sub_rng = rng

        for attempt in range(self.max_constant_tries):
            c_rng, p_rng, sub_rng = self._stream_rngs(idx, attempt, rng)
            values = self.sampler.sample(tmpl, c_rng)
            expr = tmpl.instantiate(values)
            try:
                # Overflow and invalid operations are EXPECTED here and are
                # already handled: a candidate that blows up is precisely what
                # the finite filter and `pool_is_usable` reject and redraw.
                # Left un-suppressed, NumPy warns once per lambdified template
                # per first offending draw — and because whether a template
                # overflows depends on the coefficients drawn, fresh warnings
                # keep appearing every epoch, burying warnings that matter.
                # Scoped to candidate evaluation only, so an overflow in
                # val/test or in scoring still surfaces.
                with np.errstate(over='ignore', invalid='ignore',
                                 divide='ignore'):
                    pool = sample_pool(expr, n_points, rng=p_rng)
            except RuntimeError:
                rejected += 1
                continue
            if not pool_is_usable(pool, max_abs=self.max_abs):
                rejected += 1
                continue
            return pool, expr, rejected, False, sub_rng

        # Every draw was pathological.  The reference realisation came from the
        # generator and already passed the construction probe, so it always
        # yields a usable cloud — the example degrades to baseline behaviour
        # instead of raising inside a DataLoader worker.
        expr = s['expr']
        _c, p_rng, sub_rng = self._stream_rngs(idx, self.max_constant_tries, rng)
        return sample_pool(expr, n_points, rng=p_rng), expr, rejected, True, sub_rng

    def __getitem__(self, idx):
        if not self.dynamic_constants:
            return super().__getitem__(idx)

        s = self.samples[idx]
        # None under deterministic augmentation: every stream is derived from
        # (constant_seed, stage, epoch, idx) instead, so nothing depends on
        # mutable global RNG state inside a worker.
        rng = None
        pool, _expr, rejected, fallback, sub_rng = self.draw_realization(
            idx, rng=rng)

        self.n_constant_draws += 1
        self.n_constant_rejections += rejected
        self.n_constant_fallbacks += int(fallback)

        points = subsample_and_normalize(
            pool, self.n_points, self.target_d, rng=sub_rng,
            random_subsample=True)
        return {
            'points': points,
            'input_ids': s['input_ids'],
            'attn_mask': s['attn_mask'],
        }


class DynamicConstantMultiViewDataset(DynamicConstantPointCloudDataset):
    """*n_views* subsamples of ONE dynamically-instantiated function.

    Subsample-JEPA trains the encoder to map different partial observations of
    the same function to the same latent.  Coefficients are therefore drawn
    **once per item and shared by every view**: the views must differ in which
    points were observed, not in which function was observed, or the
    consistency objective is being trained against a premise that is false.

    Composes with dynamic constants exactly as expected — each epoch the item
    is a new function, and the views are a new partial look at it:

    .. code-block:: text

        sample constants  ->  one function
              |
              +-- pool of n_points * pool_mult points
                    |
                    +-- view 0 : view_points rows
                    +-- view 1 : view_points rows      (same function)

    Every view's subsample draws from its own deterministic stream, so adding a
    view cannot shift the others.
    """

    def __init__(
        self,
        templates: Sequence[ConstantTemplate],
        tokenizer: PrefixTokenizer,
        *,
        n_views: int = 2,
        view_points: Optional[int] = None,
        pool_mult: int = 4,
        **kwargs,
    ):
        super().__init__(templates, tokenizer, pool_mult=pool_mult, **kwargs)
        self.n_views = int(n_views)
        self.view_points = int(view_points or self.n_points)

    def _view_rng(self, idx: int, view: int):
        if self.constant_seed is None:
            return None
        return np.random.RandomState(augmentation_seed(
            self.constant_seed, self.stage, self.epoch, idx, f'view{view}'))

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.dynamic_constants:
            pool, _expr, rejected, fallback, _sub = self.draw_realization(idx)
            self.n_constant_draws += 1
            self.n_constant_rejections += rejected
            self.n_constant_fallbacks += int(fallback)
        else:
            # Static control: the reference realisation, still multi-view.
            _c, p_rng, _s = self._stream_rngs(idx, 0, None)
            pool = sample_pool(s['expr'], self.n_points * self.pool_mult,
                               rng=p_rng)

        views = [
            subsample_and_normalize(
                pool, self.view_points, self.target_d,
                rng=self._view_rng(idx, v), random_subsample=True)
            for v in range(self.n_views)
        ]
        return {
            # (n_views, view_points, target_d)
            'points_views': torch.stack(views),
            # view 0 under the standard key, so single-view code still works
            'points': views[0],
            'input_ids': s['input_ids'],
            'attn_mask': s['attn_mask'],
        }


# ============================================================================
# Splits
# ============================================================================

def _canonical_groups(templates: Sequence[ConstantTemplate],
                      tokenizer: PrefixTokenizer,
                      progress: bool | str = False) -> list:
    """Group key per template: its canonical token sequence."""
    groups = []
    for t in _maybe_tqdm(templates, progress, 'grouping by canonical form'):
        try:
            groups.append(tuple(t.token_ids(tokenizer)))
        except Exception:
            groups.append(('__unparseable__', id(t)))
    return groups


def canonical_split_report(train_ds, val_ds, test_ds, strict: bool = True) -> dict:
    """Print canonical-form counts and cross-split overlaps; assert they are 0.

    The canonical form is the split unit, so an overlap would mean the model
    had already seen the held-out structure with different coefficients — an
    easier task than the generalisation to unseen structures this experiment
    claims to measure.  Checked on the built datasets rather than on the
    pre-split index arrays, so dropped expressions and dataset construction
    cannot reintroduce leakage unnoticed.
    """
    forms = {
        'train': set(train_ds.canonical_forms),
        'val': set(val_ds.canonical_forms),
        'test': set(test_ds.canonical_forms),
    }
    for name in ('train', 'val', 'test'):
        print(f'number of canonical {name} forms: {len(forms[name])}')

    overlaps = {}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        n = len(forms[a] & forms[b])
        overlaps[f'{a}/{b}'] = n
        print(f'overlap {a}/{b}: {n}')

    if strict:
        for pair, n in overlaps.items():
            if n:
                raise AssertionError(
                    f'{n} canonical form(s) appear in both {pair} splits; '
                    f'the split unit is not the canonical form.'
                )

    return {
        'n_train_forms': len(forms['train']),
        'n_val_forms': len(forms['val']),
        'n_test_forms': len(forms['test']),
        'overlaps': overlaps,
    }


def build_template_splits(
    templates: Sequence[ConstantTemplate],
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    dynamic_constants: bool = True,
    sampler: Optional[ConstantSampler] = None,
    constant_seed: Optional[int] = None,
    max_constant_tries: int = 8,
    max_abs: float = 1e6,
    cache_eval: bool = False,
    group_by_tokens: bool = True,
    progress: bool = False,
) -> tuple[DynamicConstantPointCloudDataset,
           DynamicConstantPointCloudDataset,
           DynamicConstantPointCloudDataset]:
    """Split canonical templates into train/val/test datasets.

    Uses the same ``_split_indices`` call as :func:`build_synthetic_splits`, so
    a template list built in deduplicated corpus order yields the *same*
    partition as the pre-template pipeline at the same ``seed`` — the dynamic
    run and its baseline see the same structures.

    Only the training split can carry dynamic constants; val and test are fixed
    structures with fixed coefficients and deterministic clouds.

    Returns:
        ``(train_ds, val_ds, test_ds)``
    """
    groups = (_canonical_groups(templates, tokenizer, progress=progress)
              if group_by_tokens else None)
    splits = _split_indices(
        len(templates), seed, train_frac, val_frac, groups=groups)

    # Fit the shared magnitude pool on the TRAINING forms, before any dataset
    # is built. Doing it here rather than letting the train dataset fit itself
    # keeps the fit visibly train-only even if the caller reuses this sampler.
    if sampler is not None and dynamic_constants and not sampler.is_fitted:
        sampler.fit([templates[i] for i in splits['train']])
        print(f'constant sampler fitted on {len(splits["train"])} train '
              f'forms: {sampler.describe()}')

    datasets = {}
    for name, indices in splits.items():
        is_train = name == 'train'
        datasets[name] = DynamicConstantPointCloudDataset(
            [templates[i] for i in indices], tokenizer,
            dynamic_constants=(dynamic_constants and is_train),
            sampler=sampler,
            constant_seed=constant_seed if is_train else None,
            max_constant_tries=max_constant_tries,
            max_abs=max_abs,
            n_points=n_points,
            max_seq_len=max_seq_len,
            max_vars=max_vars,
            resample=is_train,
            cache=(cache_eval and not is_train),
            progress=f'{name} point clouds' if progress else False,
        )
        mode = ('dynamic constants' if datasets[name].dynamic_constants
                else 'fixed constants')
        print(f'Templates {name}: {len(datasets[name])} forms '
              f'({datasets[name].unique_canonical_count()} unique canonical, '
              f'{mode})')

    _report_leakage(datasets)
    return datasets['train'], datasets['val'], datasets['test']


def build_multiview_template_splits(
    templates: Sequence[ConstantTemplate],
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    n_views: int = 2,
    view_points: Optional[int] = None,
    pool_mult: int = 4,
    dynamic_constants: bool = True,
    sampler: Optional[ConstantSampler] = None,
    constant_seed: Optional[int] = None,
    max_constant_tries: int = 8,
    max_abs: float = 1e6,
    cache_eval: bool = False,
    group_by_tokens: bool = True,
    progress: bool = False,
) -> tuple[DynamicConstantMultiViewDataset,
           DynamicConstantPointCloudDataset,
           DynamicConstantPointCloudDataset]:
    """Multi-view training split, deterministic single-view val/test.

    Uses the same ``_split_indices`` call as :func:`build_template_splits` and
    :func:`build_synthetic_splits`, so at a given *seed* the partition matches
    the single-view and pre-template runs exactly and results stay comparable.

    Returns:
        ``(train_ds, val_ds, test_ds)``
    """
    groups = (_canonical_groups(templates, tokenizer, progress=progress)
              if group_by_tokens else None)
    splits = _split_indices(
        len(templates), seed, train_frac, val_frac, groups=groups)

    if sampler is not None and dynamic_constants and not sampler.is_fitted:
        sampler.fit([templates[i] for i in splits['train']])
        print(f'constant sampler fitted on {len(splits["train"])} train '
              f'forms: {sampler.describe()}')

    train_ds = DynamicConstantMultiViewDataset(
        [templates[i] for i in splits['train']], tokenizer,
        n_views=n_views, view_points=view_points, pool_mult=pool_mult,
        dynamic_constants=dynamic_constants, sampler=sampler,
        constant_seed=constant_seed, max_constant_tries=max_constant_tries,
        max_abs=max_abs,
        n_points=n_points, max_seq_len=max_seq_len, max_vars=max_vars,
        resample=True,
        progress='train point clouds' if progress else False,
    )
    print(f'Templates train: {len(train_ds)} forms '
          f'({train_ds.unique_canonical_count()} unique canonical, '
          f'{n_views} views each, '
          f'{"dynamic" if train_ds.dynamic_constants else "fixed"} constants)')

    datasets = {'train': train_ds}
    for name in ('val', 'test'):
        datasets[name] = DynamicConstantPointCloudDataset(
            [templates[i] for i in splits[name]], tokenizer,
            dynamic_constants=False, sampler=sampler, constant_seed=None,
            n_points=n_points, max_seq_len=max_seq_len, max_vars=max_vars,
            resample=False, cache=cache_eval,
            progress=f'{name} point clouds' if progress else False,
        )
        print(f'Templates {name}: {len(datasets[name])} forms '
              f'({datasets[name].unique_canonical_count()} unique canonical, '
              f'fixed constants)')

    _report_leakage(datasets)
    return train_ds, datasets['val'], datasets['test']


# ============================================================================
# Diagnostics
# ============================================================================

def describe_realizations(dataset: DynamicConstantPointCloudDataset,
                          n_forms: int = 3, n_realizations: int = 3,
                          seed: int = 0, strict: bool = True) -> bool:
    """Print several coefficient realisations of a few canonical forms.

    Verifies the invariant the whole experiment rests on: the decoder target is
    a function of the structure alone.  Returns True when every realisation of
    every form produced the same token sequence.
    """
    rng = np.random.RandomState(seed)
    sampler = dataset.sampler
    tokenizer = dataset.tokenizer
    n_forms = min(n_forms, len(dataset))
    picks = rng.choice(len(dataset), size=n_forms, replace=False)

    ok = True
    for pos, idx in enumerate(sorted(int(i) for i in picks)):
        s = dataset.samples[idx]
        tmpl = s['template']
        target_ids = tuple(int(t) for t in s['input_ids'].tolist())

        print(f'\n=== canonical form {pos + 1}/{n_forms} '
              f'(index {idx}, {tmpl.n_constants} coefficient slots) ===')
        print(f'canonical target:\n    {tokenizer.decode(target_ids)}')

        for r in range(n_realizations):
            values = (tmpl.reference_constants if r == 0
                      else sampler.sample(tmpl, rng))
            expr = tmpl.instantiate(values)
            label = 'reference' if r == 0 else f'realization {r}'
            print(f'{label}:\n    {sp.sstr(expr.sympy_expr, full_prec=False)}')

            ids = tokenizer.encode(expr.prefix)
            padded = tuple(ids + [tokenizer.pad_id]
                           * (dataset.max_seq_len - len(ids)))
            if padded != target_ids:
                ok = False
                print(f'    !! token target CHANGED: {tokenizer.decode(ids)}')

    print(f'\nall realizations produce the same decoder target: {ok}')
    if strict and not ok:
        raise AssertionError(
            'Coefficient values changed the decoder token target.')
    return ok


def stage2_fingerprint(dataset, epochs: Sequence[int] = (0, 5, 15, 29),
                       indices: Optional[Sequence[int]] = None,
                       n_indices: int = 8,
                       stage: str = 'supervised') -> str:
    """Hash of the realisations *dataset* serves at fixed ``(epoch, idx)``.

    The point of decoupling the model seed from the data seed is that two runs
    differing only in pretraining must see the *same* supervised data.  That
    claim is easy to assert and easy to break silently, so it gets a fingerprint
    rather than an argument: covering the instantiated coefficients, the raw
    point cloud, the normalised tensor the encoder receives, and the target
    tokens, over several epochs at fixed sample indices.

    Depends only on the dataset's ``constant_seed`` (the data seed), never on
    the model seed or on global RNG state.  So it is equal across two arms that
    share a data seed, and different across data seeds — both worth checking.

    The dataset's epoch/stage are restored before returning, so calling this
    mid-experiment cannot perturb the run.
    """
    saved_epoch, saved_stage = dataset.epoch, dataset.stage
    if indices is None:
        step = max(1, len(dataset) // max(n_indices, 1))
        indices = [i * step for i in range(n_indices) if i * step < len(dataset)]
    indices = [int(i) for i in indices if 0 <= int(i) < len(dataset)]

    h = hashlib.sha256()
    h.update(f'{AUGMENTATION_VERSION}|{stage}|{list(epochs)}|{indices}'.encode())
    try:
        for epoch in epochs:
            dataset.set_epoch(int(epoch), stage=stage)
            for idx in indices:
                item = dataset[idx]
                h.update(np.ascontiguousarray(
                    item['points'].numpy()).tobytes())
                h.update(np.ascontiguousarray(
                    item['input_ids'].numpy()).tobytes())
                if dataset.dynamic_constants:
                    pool, expr, _rej, _fb, _sub = dataset.draw_realization(idx)
                    values = getattr(expr, 'values', None)
                    if values is not None:
                        h.update(np.asarray(values, dtype=np.float64).tobytes())
                    h.update(np.ascontiguousarray(
                        np.asarray(pool, dtype=np.float64)).tobytes())
    finally:
        dataset.epoch, dataset.stage = saved_epoch, saved_stage
    return h.hexdigest()[:16]


def audit_constant_sampling(dataset: DynamicConstantPointCloudDataset,
                            n: int = 256, seed: int = 0,
                            verbose: bool = True) -> dict:
    """Draw *n* realisations in this process and report rejection statistics.

    The dataset's own counters live in whichever process called
    ``__getitem__``; with DataLoader workers those are worker-local and the
    parent never sees them.  This runs the same draw path in the main process
    so the numbers are observable.
    """
    if not dataset.dynamic_constants:
        if verbose:
            print('constant sampling audit: dynamic_constants=False, nothing '
                  'to resample')
        return {'n_drawn': 0, 'n_rejected': 0, 'n_fallback': 0}

    rng = np.random.RandomState(seed)
    n = min(n, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)

    n_rejected = 0
    n_fallback = 0
    for idx in idxs:
        _pool, _expr, rejected, fallback, _sub = dataset.draw_realization(
            int(idx), rng=rng)
        n_rejected += rejected
        n_fallback += int(fallback)

    stats = {
        'n_drawn': int(n),
        'n_rejected': int(n_rejected),
        'n_fallback': int(n_fallback),
        'rejections_per_example': n_rejected / max(n, 1),
        'fallback_rate': n_fallback / max(n, 1),
    }
    if verbose:
        print(f'constant sampling audit over {n} examples: '
              f'{n_rejected} rejected draws '
              f'({stats["rejections_per_example"]:.3f} per accepted example), '
              f'{n_fallback} fell back to reference coefficients '
              f'({100 * stats["fallback_rate"]:.1f}%)')
    return stats
