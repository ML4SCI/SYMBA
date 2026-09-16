"""Per-example agreement between two runs.

A difference in aggregate scores says an arm is better; it does not say *how*.
Two arms at 35.4 % and 36.1 % equivalence could mean:

* **nested** — the better arm solves everything the baseline solves, plus a few
  more.  That is a clean capability gain;
* **rotated** — the arms solve substantially different sets of the same size,
  and the aggregate difference is the residue of a much larger churn.  Then the
  headline number understates the variance and a paired mean is misleading.

The second case is common and invisible in a summary table, so this module works
from the per-example ``details`` that ``evaluate_predictions`` already writes
into every ``metrics.json``.  Nothing has to be re-run.

The test is McNemar's: only the **discordant** examples carry information about
which arm is better, because a problem both arms solve (or both miss) says
nothing about the difference between them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    'AgreementTable',
    'agreement',
    'load_details',
    'mcnemar_exact',
    'compare_runs',
    'load_run_matrix',
    'paired_seed_report',
    'stratified_agreement',
]


def load_details(metrics_path) -> list[dict]:
    """Per-example records from a run's ``metrics.json``.

    Ordered by test-set index, because ``evaluate_predictions`` requires the
    eval loader to be built with ``shuffle=False``.  That is what makes
    element-wise comparison between two runs meaningful.
    """
    with open(metrics_path) as f:
        payload = json.load(f)
    details = payload.get('details')
    if details is None:
        raise ValueError(f'{metrics_path} has no per-example "details"')
    return details


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for a discordant split.

    Under the null that the two arms are equally likely to win a discordant
    example, ``only_b ~ Binomial(only_a + only_b, 0.5)``.  The exact form is
    used rather than the chi-squared approximation because the discordant
    counts here are small enough for it to matter.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(only_b, n, 0.5).pvalue)
    except ImportError:
        from math import comb
        k = min(only_a, only_b)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return float(min(1.0, 2 * tail))


class AgreementTable:
    """2x2 contingency between two arms on one binary metric."""

    def __init__(self, both: int, only_a: int, only_b: int, neither: int,
                 name_a: str = 'A', name_b: str = 'B', metric: str = 'equiv'):
        self.both = both
        self.only_a = only_a
        self.only_b = only_b
        self.neither = neither
        self.name_a = name_a
        self.name_b = name_b
        self.metric = metric

    @property
    def n(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither

    @property
    def a_correct(self) -> int:
        return self.both + self.only_a

    @property
    def b_correct(self) -> int:
        return self.both + self.only_b

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    @property
    def is_nested(self) -> bool:
        """True when B solves everything A solves — a pure capability gain."""
        return self.only_a == 0

    @property
    def churn(self) -> float:
        """Discordant examples as a fraction of the *net* gain.

        1.0 means every example B gained was a genuine addition.  A large value
        means the arms are trading examples and the net difference is a small
        residue of a much bigger reshuffle — so the aggregate gap is a weak
        summary of what changed.
        """
        net = self.only_b - self.only_a
        if net == 0:
            return float('inf') if self.discordant else 0.0
        return self.discordant / abs(net)

    def pvalue(self) -> float:
        return mcnemar_exact(self.only_a, self.only_b)

    def to_dict(self) -> dict:
        return {
            'metric': self.metric, 'n': self.n,
            'both': self.both, 'only_a': self.only_a,
            'only_b': self.only_b, 'neither': self.neither,
            'a_correct': self.a_correct, 'b_correct': self.b_correct,
            'net': self.only_b - self.only_a,
            'discordant': self.discordant,
            'nested': self.is_nested,
            'churn': self.churn,
            'pvalue': self.pvalue(),
        }

    def report(self) -> str:
        net = self.only_b - self.only_a
        lines = [
            f'{self.metric}: {self.name_a} {self.a_correct}/{self.n} '
            f'({100 * self.a_correct / self.n:.2f}%) vs '
            f'{self.name_b} {self.b_correct}/{self.n} '
            f'({100 * self.b_correct / self.n:.2f}%)',
            f'  both correct        : {self.both}',
            f'  only {self.name_a:<14}: {self.only_a}',
            f'  only {self.name_b:<14}: {self.only_b}',
            f'  neither             : {self.neither}',
            f'  net                 : {net:+d}',
            f'  discordant          : {self.discordant} '
            f'({100 * self.discordant / self.n:.2f}% of the test set)',
            f'  McNemar exact p     : {self.pvalue():.4g}',
        ]
        if self.is_nested:
            lines.append(f'  -> NESTED: {self.name_b} solves everything '
                         f'{self.name_a} solves, plus {self.only_b}.')
        else:
            lines.append(
                f'  -> NOT nested: {self.name_b} lost {self.only_a} that '
                f'{self.name_a} solved. Churn = {self.churn:.1f}x the net gain, '
                f'so the arms differ on {self.discordant} examples to move '
                f'{net:+d}.')
        return '\n'.join(lines)


def _flags(details: Sequence[dict], metric: str) -> np.ndarray:
    if metric == 'r2>0.9':
        return np.array([
            d.get('r2') is not None and np.isfinite(d['r2']) and d['r2'] > 0.9
            for d in details], dtype=bool)
    return np.array([bool(d.get(metric)) for d in details], dtype=bool)


def agreement(details_a: Sequence[dict], details_b: Sequence[dict],
              metric: str = 'equiv', name_a: str = 'A',
              name_b: str = 'B') -> AgreementTable:
    """Contingency between two runs on *metric*.

    ``metric`` is ``'equiv'``, ``'exact'``, or ``'r2>0.9'``.  Both runs must
    have been scored on the same test set in the same order.
    """
    if len(details_a) != len(details_b):
        raise ValueError(
            f'{len(details_a)} vs {len(details_b)} examples; the two runs were '
            f'not scored on the same test set')

    gt_a = [d.get('gt') for d in details_a]
    gt_b = [d.get('gt') for d in details_b]
    if gt_a != gt_b:
        n_diff = sum(1 for x, y in zip(gt_a, gt_b) if x != y)
        raise ValueError(
            f'ground truth differs on {n_diff} examples; the runs used '
            f'different splits or orderings and cannot be paired')

    a = _flags(details_a, metric)
    b = _flags(details_b, metric)
    return AgreementTable(
        both=int((a & b).sum()), only_a=int((a & ~b).sum()),
        only_b=int((~a & b).sum()), neither=int((~a & ~b).sum()),
        name_a=name_a, name_b=name_b, metric=metric)


def stratified_agreement(details_a, details_b, metric: str = 'equiv',
                         bins: Sequence[int] = (0, 10, 20, 30, 100)) -> dict:
    """Where the gains and losses sit, by ground-truth expression length.

    A gain concentrated on short expressions means something different from one
    spread evenly: the former is the model getting easy structures right, the
    latter a broad improvement.
    """
    lengths = np.array([len(str(d.get('gt', '')).split()) for d in details_a])
    a = _flags(details_a, metric)
    b = _flags(details_b, metric)

    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lengths >= lo) & (lengths < hi)
        if not m.any():
            continue
        out[f'{lo}-{hi}'] = {
            'n': int(m.sum()),
            'a_correct': int((a & m).sum()),
            'b_correct': int((b & m).sum()),
            'only_a': int((a & ~b & m).sum()),
            'only_b': int((~a & b & m).sum()),
        }
    return out


def compare_runs(runs_a: dict, runs_b: dict, metric: str = 'equiv',
                 name_a: str = 'pretrain_0', name_b: str = 'pretrain_10',
                 verbose: bool = True) -> dict:
    """Pair two arms seed-by-seed and pool the discordant counts.

    Args:
        runs_a, runs_b: ``{seed: details}``, e.g. from :func:`load_details`.
        metric: ``'equiv'``, ``'exact'`` or ``'r2>0.9'``.

    Pooling the 2x2 counts across seeds is the right aggregation here: each seed
    contributes its own discordant examples, and McNemar over the pool asks
    whether *B* wins discordant examples more often than chance overall.
    Per-seed tables are reported too, because a consistent direction across
    seeds is stronger evidence than one pooled p-value.
    """
    seeds = sorted(set(runs_a) & set(runs_b))
    if not seeds:
        raise ValueError('no seeds in common between the two arms')

    per_seed = {}
    tot = dict(both=0, only_a=0, only_b=0, neither=0)
    for s in seeds:
        t = agreement(runs_a[s], runs_b[s], metric=metric,
                      name_a=name_a, name_b=name_b)
        per_seed[s] = t
        for k in tot:
            tot[k] += getattr(t, k)

    pooled = AgreementTable(**tot, name_a=name_a, name_b=name_b, metric=metric)

    if verbose:
        print(f'=== {metric}: {name_a} vs {name_b} over {len(seeds)} seeds ===\n')
        print(f'{"seed":>8}{"both":>8}{"only_a":>9}{"only_b":>9}'
              f'{"net":>7}{"nested":>9}{"p":>10}')
        for s in seeds:
            t = per_seed[s]
            print(f'{s:>8}{t.both:>8}{t.only_a:>9}{t.only_b:>9}'
                  f'{t.only_b - t.only_a:>+7d}{str(t.is_nested):>9}'
                  f'{t.pvalue():>10.3g}')
        wins = sum(1 for s in seeds if per_seed[s].only_b > per_seed[s].only_a)
        nested = sum(1 for s in seeds if per_seed[s].is_nested)
        print(f'\n{name_b} wins the discordant count in {wins}/{len(seeds)} '
              f'seeds; nested in {nested}/{len(seeds)}\n')
        print('POOLED')
        print(pooled.report())

    return {'per_seed': per_seed, 'pooled': pooled, 'seeds': seeds}


def load_run_matrix(root, pretrain_values: Sequence[int],
                    seeds: Sequence,
                    metrics_name: str = 'supervised/metrics.json',
                    subdir=None) -> dict:
    """``{pretrain_epochs: {seed: details}}`` from a checkpoint tree.

    Skips runs whose ``metrics.json`` is missing, so a partially finished sweep
    still analyses.

    Args:
        subdir: Maps a *seeds* entry to its directory name.  Defaults to
            ``seed_{s}``.  An experiment that separates model and data seeds
            lays runs out differently, so the naming has to be injectable
            rather than assumed.
    """
    root = Path(root)
    name = subdir or (lambda s: f'seed_{s}')
    out: dict = {}
    for pe in pretrain_values:
        out[pe] = {}
        for seed in seeds:
            path = root / f'pretrain_{pe}' / name(seed) / metrics_name
            if not path.exists():
                continue
            try:
                out[pe][seed] = load_details(path)
            except (ValueError, json.JSONDecodeError) as e:
                print(f'[agreement] skipping {path}: {e}')
    return out


def paired_seed_report(values_a: dict, values_b: dict, label: str = 'metric',
                       name_a: str = 'A', name_b: str = 'B',
                       verbose: bool = True) -> dict:
    """Seed-level paired comparison of a scalar metric.

    This is the primary analysis, and it is **not** interchangeable with a
    pooled McNemar over examples.  McNemar asks whether B wins discordant
    *examples*; this asks whether B wins across independent *training runs*,
    which is the question when the thing being compared is a training
    procedure.  A pooled example-level test can be significant while the effect
    fails to reproduce across seeds — the seeds are the unit of replication.

    Args:
        values_a, values_b: ``{seed: value}``, in the same units in and out.

    Returns a dict with the paired mean, its 95% CI, both test p-values, and
    the improved/tied/worsened split.
    """
    seeds = sorted(set(values_a) & set(values_b))
    if not seeds:
        raise ValueError('no seeds in common between the two arms')

    a = np.array([values_a[s] for s in seeds], dtype=float)
    b = np.array([values_b[s] for s in seeds], dtype=float)
    d = b - a
    n = len(d)

    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0

    t_p = wil_p = float('nan')
    if n > 1 and sd > 0:
        from scipy import stats
        t_p = float(stats.ttest_rel(b, a).pvalue)
        # Wilcoxon is undefined when every difference is zero, and warns when
        # n is small; it is reported alongside t because a single outlying seed
        # can carry a t-test that the rank test will not.
        try:
            wil_p = float(stats.wilcoxon(b, a).pvalue)
        except ValueError:
            wil_p = float('nan')
        ci = 1.96 * se
    else:
        ci = 0.0

    out = {
        'label': label, 'seeds': seeds, 'n': n,
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'paired_mean': mean, 'sd': sd, 'se': se,
        'ci95': (mean - ci, mean + ci),
        'ttest_p': t_p, 'wilcoxon_p': wil_p,
        'improved': int((d > 0).sum()),
        'tied': int((d == 0).sum()),
        'worsened': int((d < 0).sum()),
        'per_seed': {s: float(x) for s, x in zip(seeds, d)},
    }

    if verbose:
        print(f'{label}: {name_a} {a.mean():.3f} -> {name_b} {b.mean():.3f}')
        print(f'  paired mean  : {mean:+.3f}  (sd {sd:.3f}, n={n})')
        print(f'  95% CI       : [{out["ci95"][0]:+.3f}, {out["ci95"][1]:+.3f}]')
        print(f'  paired t     : p={t_p:.4g}')
        print(f'  Wilcoxon     : p={wil_p:.4g}')
        print(f'  improved/tied/worsened: {out["improved"]}/{out["tied"]}'
              f'/{out["worsened"]}')
    return out
