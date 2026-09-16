"""
PyTorch Dataset for symbolic regression.

Wraps Expression objects into point-cloud + token-sequence pairs.
Each __getitem__ resamples the point cloud for data augmentation.
"""

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from symbolic_jepa.expressions import Expression, load_feynman_csv
from symbolic_jepa.tokenizer import PrefixTokenizer

# Fixed seed for deterministic eval point clouds, independent of model seed.
_EVAL_DATA_SEED = 999_999_937


def _maybe_tqdm(iterable, progress, desc: str, total: int | None = None):
    """Wrap *iterable* in a tqdm bar when *progress* is truthy.

    *progress* may be a bool or a string; a string replaces *desc*, so callers
    can label several bars without threading an extra argument through.
    """
    if not progress:
        return iterable
    if isinstance(progress, str):
        desc = progress
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    return tqdm(iterable, desc=desc, total=total, leave=False)


def sample_pool(
    expr: Expression,
    n_points: int,
    rng: np.random.RandomState | None = None,
    min_finite: int = 10,
    n_tries: int = 3,
) -> np.ndarray:
    """Sample *n_points* from *expr* and keep the finite rows.

    Args:
        expr: Expression to sample from.
        n_points: Number of points to draw, before finite filtering.
        rng: RandomState for reproducible sampling, or None for the global
            numpy RNG.
        min_finite: Minimum finite rows required.
        n_tries: Redraws allowed before giving up.

    Returns:
        (n_finite, n_vars + 1) array with n_finite >= min_finite.

    Raises:
        RuntimeError: If fewer than *min_finite* finite points survive
            filtering on every attempt.
    """
    src_rng = rng if rng is not None else np.random
    cloud = np.empty((0, 0))

    for attempt in range(n_tries):
        draw_rng = src_rng if attempt == 0 else np.random.RandomState(
            int(src_rng.randint(0, 2 ** 31 - 1))
        )
        cloud = expr.sample(n_points, method='uniform', rng=draw_rng)

        # Filter non-finite rows
        finite_mask = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite_mask]

        if len(cloud) >= min_finite:
            return cloud

    raise RuntimeError(
        f"Expression {expr!r} produced fewer than {min_finite} finite points "
        f"({len(cloud)} of {n_points}) on {n_tries} attempts. Cannot build a "
        f"valid point cloud — this expression should be dropped."
    )


def subsample_and_normalize(
    pool: np.ndarray,
    n_points: int,
    target_d: int,
    rng: np.random.RandomState | None = None,
    random_subsample: bool = True,
) -> torch.Tensor:
    """Take *n_points* rows out of *pool*, then normalize and pad.

    Args:
        pool: (n_finite, n_vars + 1) array from sample_pool.
        n_points: Target number of points.
        target_d: Target feature width (max_vars + 1).
        rng: RandomState for reproducible subsampling, or None for the global
            numpy RNG.
        random_subsample: When the pool holds more points than n_points,
            take a random subset (True) or the first n_points (False).

    Returns:
        (n_points, target_d) float32 tensor.
    """
    src_rng = rng if rng is not None else np.random
    cloud = pool

    # Pad or truncate to n_points
    if len(cloud) >= n_points:
        if random_subsample:
            choice_idx = src_rng.choice(len(cloud), n_points, replace=False)
            cloud = cloud[choice_idx]
        else:
            cloud = cloud[:n_points]
    else:
        # Repeat existing points to fill to n_points
        n_need = n_points - len(cloud)
        extra_idx = src_rng.choice(len(cloud), n_need, replace=True)
        cloud = np.vstack([cloud, cloud[extra_idx]])

    # Normalize per-column
    cloud = (cloud - cloud.mean(axis=0)) / (cloud.std(axis=0) + 1e-8)

    # Pad dimensions to target_d
    _, d = cloud.shape
    if d < target_d:
        cloud = np.pad(cloud, ((0, 0), (0, target_d - d)))
    else:
        cloud = cloud[:, :target_d]

    return torch.tensor(cloud, dtype=torch.float32)


def sample_and_normalize(
    expr: Expression,
    n_points: int,
    target_d: int,
    rng: np.random.RandomState | None = None,
    random_subsample: bool = True,
    pool_mult: int = 1,
) -> torch.Tensor:
    """Sample a point cloud from *expr*, then filter, resize, normalize, pad.

    Args:
        expr: Expression to sample from.
        n_points: Target number of points.
        target_d: Target feature width (max_vars + 1).
        rng: RandomState for reproducible sampling, or None for the global
            numpy RNG.
        random_subsample: When more finite points survive than n_points,
            take a random subset (True) or the first n_points (False).
        pool_mult: Draw n_points * pool_mult candidates before subsampling.
            At pool_mult=1 the subsample is only a permutation.

    Returns:
        (n_points, target_d) float32 tensor.

    Raises:
        RuntimeError: If fewer than 10 finite points survive filtering.
    """
    pool = sample_pool(expr, n_points * pool_mult, rng=rng)
    return subsample_and_normalize(
        pool, n_points, target_d, rng=rng, random_subsample=random_subsample,
    )


class PointCloudDataset(Dataset):
    """Dataset of (point_cloud, token_ids) pairs.

    When resample=True (train), each __getitem__ freshly samples the point
    cloud, providing infinite augmentation.  When resample=False (val/test),
    point clouds are deterministic: seeded by a fixed data seed + sample index.
    """

    def __init__(
        self,
        expressions: list[Expression],
        tokenizer: PrefixTokenizer,
        n_points: int = 1000,
        max_seq_len: int = 64,
        max_vars: int = 9,
        resample: bool = True,
        cache: bool = False,
        progress: bool | str = False,
    ):
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.max_vars = max_vars
        self.target_d = max_vars + 1  # input vars + output
        self.resample = resample
        # Only meaningful when resample=False: those clouds are deterministic,
        # so re-deriving them every epoch is pure waste.  Caching lets an
        # eval loader run with num_workers=0 without paying for it.
        self.cache = cache and not resample
        self._cloud_cache: dict[int, torch.Tensor] = {}

        # Pre-tokenize and filter
        self.samples: list[dict] = []
        n_dropped_cloud = 0
        # The probe below samples n_points from every expression (one lambdify
        # each), so this loop dominates dataset construction on large sets.
        for expr in _maybe_tqdm(expressions, progress, 'building point clouds'):
            try:
                ids = expr.tokenize(tokenizer)
            except Exception:
                continue

            if len(ids) > max_seq_len or tokenizer.unk_id in ids:
                continue

            # Validate that the expression can produce enough finite points.
            # Use a deterministic probe seed so this check is reproducible.
            if not self._probe_valid(expr, n_points):
                n_dropped_cloud += 1
                continue

            pad = max_seq_len - len(ids)
            self.samples.append({
                'expr': expr,
                'token_key': tuple(ids),
                'input_ids': torch.tensor(ids + [tokenizer.pad_id] * pad, dtype=torch.long),
                'attn_mask': torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
            })

        if n_dropped_cloud > 0:
            import warnings
            warnings.warn(
                f"Dropped {n_dropped_cloud} expression(s) that could not "
                f"produce >= 10 finite points."
            )

    @property
    def token_keys(self) -> list[tuple]:
        """Prefix token sequence of every sample, in dataset order."""
        return [s['token_key'] for s in self.samples]

    def unique_sequence_count(self) -> int:
        """Number of distinct token sequences in the dataset."""
        return len(set(self.token_keys))

    @staticmethod
    def _probe_valid(expr: Expression, n_points: int, n_tries: int = 3) -> bool:
        """Return True if *expr* can produce >= 10 finite points.

        Retries up to *n_tries* times with different deterministic seeds.
        """
        for attempt in range(n_tries):
            rng = np.random.RandomState(seed=_EVAL_DATA_SEED + attempt)
            cloud = expr.sample(n_points, method='uniform', rng=rng)
            finite = np.isfinite(cloud).all(axis=1).sum()
            if finite >= 10:
                return True
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        points = self._sample_points(s['expr'], idx)
        return {
            'points': points,
            'input_ids': s['input_ids'],
            'attn_mask': s['attn_mask'],
        }

    def _sample_points(self, expr: Expression, idx: int) -> torch.Tensor:
        """Sample, normalize, and pad a point cloud from the expression.

        When resample=False (val/test), uses a deterministic RNG seeded by
        _EVAL_DATA_SEED + idx so repeated access gives the same cloud and
        the model training seed has no effect.
        """
        if self.cache and idx in self._cloud_cache:
            return self._cloud_cache[idx]

        if not self.resample:
            rng = np.random.RandomState(_EVAL_DATA_SEED + idx)
        else:
            rng = None  # use global numpy RNG (stochastic)

        cloud = sample_and_normalize(
            expr, self.n_points, self.target_d,
            rng=rng, random_subsample=self.resample,
        )
        if self.cache:
            self._cloud_cache[idx] = cloud
        return cloud


class MultiViewPointCloudDataset(PointCloudDataset):
    """Dataset returning *n_views* subsamples of one pool per equation.

    Used for subsample-JEPA: the encoder should map different numerical
    observations of the same function to similar latents.

    A pool of ``n_points * pool_mult`` points is drawn per item and each view
    takes ``view_points`` rows out of it, so the views are genuinely partial
    observations rather than independent full draws.

    View seeds derive deterministically from
    ``(train_view_seed, epoch, idx, view_idx)``, so:

    * the same expression gets fresh views every epoch;
    * runs that share data seeds but differ in lambda see identical views;
    * DataLoader worker assignment does not affect the generated views.

    The pool uses ``view_idx = -1``.

    .. warning::
       ``self.epoch`` is read inside ``__getitem__``.  Build the training
       DataLoader with ``persistent_workers=False`` — persistent workers hold
       a forked copy of this object and would never observe epoch updates
       made in the parent process, silently freezing the views.
    """

    def __init__(
        self,
        expressions: list[Expression],
        tokenizer: PrefixTokenizer,
        n_points: int = 1000,
        max_seq_len: int = 64,
        max_vars: int = 9,
        n_views: int = 2,
        train_view_seed: int = 1729,
        pool_mult: int = 4,
        view_points: int | None = None,
        progress: bool | str = False,
    ):
        super().__init__(
            expressions, tokenizer,
            n_points=n_points, max_seq_len=max_seq_len,
            max_vars=max_vars, resample=True, progress=progress,
        )
        self.n_views = n_views
        self.train_view_seed = train_view_seed
        self.pool_mult = pool_mult
        self.view_points = view_points or n_points
        self.epoch = 0  # set by the training loop before each epoch

    @staticmethod
    def view_seed(train_view_seed: int, epoch: int, idx: int, view_idx: int) -> int:
        """Deterministic 32-bit seed for one (epoch, expression, view) triple.

        Uses SHA-256 rather than Python's ``hash()``, which is randomized
        per process for strings and would break reproducibility.
        """
        key = f'{train_view_seed}|{epoch}|{idx}|{view_idx}'.encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')

    def __getitem__(self, idx):
        s = self.samples[idx]

        pool = sample_pool(
            s['expr'], self.n_points * self.pool_mult,
            rng=np.random.RandomState(
                self.view_seed(self.train_view_seed, self.epoch, idx, -1)
            ),
        )

        views = [
            subsample_and_normalize(
                pool, self.view_points, self.target_d,
                rng=np.random.RandomState(
                    self.view_seed(self.train_view_seed, self.epoch, idx, v)
                ),
                random_subsample=True,
            )
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


def _split_indices(
    n: int, seed: int, train_frac: float, val_frac: float,
    groups: list | None = None,
) -> dict[str, np.ndarray]:
    """Shuffle 0..n-1 and cut into train/val/test index arrays.

    When *groups* is given, items sharing a group key stay in one split.
    """
    rng = np.random.default_rng(seed)

    if groups is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        return {
            'train': idx[:n_train],
            'val': idx[n_train:n_train + n_val],
            'test': idx[n_train + n_val:],
        }

    buckets: dict = {}
    for i, g in enumerate(groups):
        buckets.setdefault(g, []).append(i)

    keys = list(buckets)
    order = rng.permutation(len(keys))

    # Whole buckets only, so the realised fractions drift from the requested
    # ones when buckets are large.
    n_train_target = int(train_frac * n)
    n_val_target = int(val_frac * n)
    out: dict[str, list] = {'train': [], 'val': [], 'test': []}
    for k in order:
        members = buckets[keys[k]]
        if len(out['train']) < n_train_target:
            out['train'].extend(members)
        elif len(out['val']) < n_val_target:
            out['val'].extend(members)
        else:
            out['test'].extend(members)

    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


def _token_groups(
    expressions: list[Expression], tokenizer: PrefixTokenizer,
    progress: bool | str = False,
) -> list:
    """Group key per expression: its prefix token sequence."""
    groups = []
    for e in _maybe_tqdm(expressions, progress, 'grouping by token sequence'):
        try:
            groups.append(tuple(e.tokenize(tokenizer)))
        except Exception:
            groups.append(('__unparseable__', id(e)))
    return groups


def _report_leakage(datasets: dict):
    """Print how many held-out token sequences also occur in train."""
    train_keys = set(datasets['train'].token_keys)
    for name in ('val', 'test'):
        keys = datasets[name].token_keys
        if not keys:
            continue
        dup = sum(1 for k in keys if k in train_keys)
        print(f'  leakage {name}: {dup}/{len(keys)} '
              f'({100 * dup / len(keys):.1f}%) sequences also in train')


def build_feynman_splits(
    csv_path: str,
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    group_by_tokens: bool = True,
    progress: bool = False,
) -> tuple[PointCloudDataset, PointCloudDataset, PointCloudDataset]:
    """Load Feynman equations and split into train/val/test datasets.

    Args:
        group_by_tokens: Keep equations sharing a prefix token sequence in
            the same split.
        progress: Show tqdm bars for the grouping and point-cloud passes.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    all_exprs = load_feynman_csv(csv_path)
    groups = (_token_groups(all_exprs, tokenizer, progress=progress)
              if group_by_tokens else None)
    splits = _split_indices(len(all_exprs), seed, 0.8, 0.1, groups=groups)

    datasets = {}
    for name, indices in splits.items():
        exprs = [all_exprs[i] for i in indices]
        datasets[name] = PointCloudDataset(
            exprs, tokenizer,
            n_points=n_points,
            max_seq_len=max_seq_len,
            max_vars=max_vars,
            resample=(name == 'train'),
            progress=f'{name} point clouds' if progress else False,
        )
        print(f'Feynman {name}: {len(datasets[name])} equations '
              f'({datasets[name].unique_sequence_count()} unique sequences)')

    _report_leakage(datasets)
    return datasets['train'], datasets['val'], datasets['test']


def build_synthetic_splits(
    expressions: list[Expression],
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    cache_eval: bool = False,
    group_by_tokens: bool = True,
    progress: bool = False,
) -> tuple[PointCloudDataset, PointCloudDataset, PointCloudDataset]:
    """Split synthetic expressions into train/val/test datasets.

    Args:
        cache_eval: cache the val/test clouds.  They are deterministic, so
            re-deriving them every epoch is wasted work; caching lets an eval
            loader run with num_workers=0 at no cost.  Off by default so
            existing callers are unaffected.
        group_by_tokens: keep expressions sharing a prefix token sequence in
            one split, so no held-out sequence appears verbatim in train.
        progress: show tqdm bars for the grouping and point-cloud passes.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    groups = (_token_groups(expressions, tokenizer, progress=progress)
              if group_by_tokens else None)
    splits = _split_indices(
        len(expressions), seed, train_frac, val_frac, groups=groups)

    datasets = {}
    for name, indices in splits.items():
        exprs = [expressions[i] for i in indices]
        datasets[name] = PointCloudDataset(
            exprs, tokenizer,
            n_points=n_points,
            max_seq_len=max_seq_len,
            max_vars=max_vars,
            resample=(name == 'train'),
            cache=(cache_eval and name != 'train'),
            progress=f'{name} point clouds' if progress else False,
        )
        print(f'Synthetic {name}: {len(datasets[name])} equations '
              f'({datasets[name].unique_sequence_count()} unique sequences)')

    _report_leakage(datasets)
    return datasets['train'], datasets['val'], datasets['test']


def build_multiview_synthetic_splits(
    expressions: list[Expression],
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    n_views: int = 2,
    train_view_seed: int = 1729,
    pool_mult: int = 4,
    view_points: int | None = None,
    group_by_tokens: bool = True,
    progress: bool = False,
) -> tuple[MultiViewPointCloudDataset, PointCloudDataset, PointCloudDataset]:
    """Split synthetic expressions, with a multi-view training set.

    Uses the same split logic as build_synthetic_splits, so for a given *seed*
    and *group_by_tokens* the train/val/test partition matches the single-view
    sweep exactly and results stay comparable.  Val and test remain
    deterministic single-view.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    groups = (_token_groups(expressions, tokenizer, progress=progress)
              if group_by_tokens else None)
    splits = _split_indices(
        len(expressions), seed, train_frac, val_frac, groups=groups)

    train_ds = MultiViewPointCloudDataset(
        [expressions[i] for i in splits['train']], tokenizer,
        n_points=n_points, max_seq_len=max_seq_len, max_vars=max_vars,
        n_views=n_views, train_view_seed=train_view_seed,
        pool_mult=pool_mult, view_points=view_points,
        progress='train point clouds' if progress else False,
    )
    print(f'Synthetic train: {len(train_ds)} equations '
          f'({train_ds.unique_sequence_count()} unique sequences, '
          f'{n_views} views each)')

    eval_ds = {'train': train_ds}
    for name in ('val', 'test'):
        eval_ds[name] = PointCloudDataset(
            [expressions[i] for i in splits[name]], tokenizer,
            n_points=n_points, max_seq_len=max_seq_len, max_vars=max_vars,
            resample=False, cache=True,
            progress=f'{name} point clouds' if progress else False,
        )
        print(f'Synthetic {name}: {len(eval_ds[name])} equations '
              f'({eval_ds[name].unique_sequence_count()} unique sequences)')

    _report_leakage(eval_ds)
    return train_ds, eval_ds['val'], eval_ds['test']
