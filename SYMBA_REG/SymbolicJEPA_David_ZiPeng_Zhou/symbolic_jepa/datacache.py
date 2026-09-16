"""Content-addressed cache for derived dataset artifacts.

Parsing the synthetic pickle is the slowest step in the pipeline: every raw
string goes through ``sympify`` and prefix conversion at a few milliseconds
each, so 200 000 strings cost 10-15 minutes — paid again on every fresh Colab
runtime, and again by the diagnostics notebook.

This module caches the two expensive derived artifacts:

* the deduplicated ``Expression`` list from :func:`load_synthetic_pkl`,
* the branching prefix tree from :func:`build_prefix_tree`.

Cache entries are **content-addressed**: the filename embeds a hash of every
input that determines the result, including the source of the library modules
that produce it.  A stale entry therefore cannot be read — a changed config or
a changed parser simply misses and rebuilds under a new name.

Unlike a checkpoint, a cache entry is *derived* data.  Losing one costs time,
not results, so reads that fail are treated as a miss and rebuilt rather than
raising.  Writes are still atomic (temp file, fsync, ``os.replace``), so a
crash mid-write cannot leave a truncated entry in place of a good one.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from symbolic_jepa import dataset as _dataset_mod
from symbolic_jepa import evaluation as _evaluation_mod
from symbolic_jepa import expressions as _expressions_mod
from symbolic_jepa import tokenizer as _tokenizer_mod
from symbolic_jepa.checkpointing import backup_file_to_drive
from symbolic_jepa.evaluation import build_prefix_tree
from symbolic_jepa.expressions import load_synthetic_pkl

__all__ = [
    'DataCache',
    'cached_synthetic_expressions',
    'cached_prefix_tree',
    'expressions_cache_key',
    'prefix_tree_cache_key',
    'module_fingerprint',
]

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def module_fingerprint(*modules) -> str:
    """Short hash of the source of *modules*.

    Included in every cache key so that editing a parser, tokenizer, or split
    rule invalidates the entries it could have changed, without anyone having
    to remember to bump a version number.
    """
    h = hashlib.sha256()
    for m in modules:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()[:12]


def _hash_parts(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b'\0')
    return h.hexdigest()[:16]


def _hash_token_keys(token_keys) -> str:
    """Exact hash of a list of token-ID sequences."""
    h = hashlib.sha256()
    for seq in token_keys:
        h.update(np.asarray(seq, dtype=np.int64).tobytes())
        h.update(b'\n')
    return h.hexdigest()[:16]


class DataCache:
    """Local-first store for derived artifacts, with a Drive copy.

    Mirrors the checkpoint architecture: the local filesystem is authoritative
    during a live runtime and Drive is a durable copy that survives the runtime
    being destroyed.  Drive failures are warnings — a cache is an optimisation.

    Args:
        local_root: Directory for cache entries, e.g. ``/content/..._cache``.
        drive_root: Optional Drive directory for durable copies.
        enabled: Set False to bypass the cache entirely and always rebuild.
        verbose: Print hit/miss/store lines.
    """

    def __init__(self, local_root, drive_root=None, enabled: bool = True,
                 verbose: bool = True):
        self.local_root = Path(local_root)
        self.drive_root = None if drive_root is None else Path(drive_root)
        self.enabled = enabled
        self.verbose = verbose
        if self.enabled:
            self.local_root.mkdir(parents=True, exist_ok=True)

    def _paths(self, name: str, key: str):
        fname = f'{name}-{key}.pkl'
        return (self.local_root / fname,
                None if self.drive_root is None else self.drive_root / fname)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _read(self, path: Path, key: str):
        """Return the payload, or None if the entry is unusable."""
        try:
            with open(path, 'rb') as f:
                entry = pickle.load(f)
            if entry.get('key') != key:
                raise ValueError('key mismatch inside cache entry')
            return entry
        except Exception as e:
            # Derived data: a bad entry is a miss, not a failure.
            print(f'[cache WARNING] ignoring unusable entry {path.name}: '
                  f'{type(e).__name__}: {e}')
            return None

    def _write(self, path: Path, entry: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.tmp-{os.getpid()}')
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(entry, f, protocol=_PICKLE_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:
            Path(tmp).unlink(missing_ok=True)
            print(f'[cache WARNING] could not store {path.name}: '
                  f'{type(e).__name__}: {e}')

    def put(self, name: str, key: str, payload, label: str | None = None) -> bool:
        """Store an already-built artifact under (*name*, *key*).

        Use this to bank an artifact a live session has already computed —
        `get_or_build` would rebuild it from scratch, which is exactly the cost
        being avoided.  Runs even when the cache is disabled for reads.

        Returns:
            True if the entry landed on the local filesystem.
        """
        local, drive = self._paths(name, key)
        self._write(local, {
            'key': key,
            'name': name,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'payload': payload,
        })
        if not local.exists():
            return False
        self._say(f'[cache] stored {local.name} '
                  f'({local.stat().st_size / 1e6:.1f} MB)'
                  + (f'  <- {label}' if label else ''))
        if drive is not None:
            backup_file_to_drive(local, drive, verbose=self.verbose)
        return True

    def get_or_build(self, name: str, key: str, build, label: str | None = None):
        """Return the cached artifact for (*name*, *key*), building it if absent.

        Lookup order is local, then Drive (copied down first), then *build*.
        """
        label = label or name
        if not self.enabled:
            self._say(f'[cache] disabled — building {label}')
            return build()

        local, drive = self._paths(name, key)

        if local.exists():
            entry = self._read(local, key)
            if entry is not None:
                self._say(f'[cache] hit  {label}  ({local.name}, '
                          f'built {entry.get("created", "?")})')
                return entry['payload']

        if drive is not None:
            try:
                available = drive.exists()
            except OSError as e:
                print(f'[cache WARNING] cannot reach Drive: '
                      f'{type(e).__name__}: {e}')
                available = False
            if available:
                tmp = local.with_name(f'{local.name}.restore-{os.getpid()}')
                try:
                    shutil.copy2(drive, tmp)
                    entry = self._read(Path(tmp), key)
                    if entry is None:
                        raise ValueError('Drive entry unusable')
                    os.replace(tmp, local)
                except Exception as e:
                    Path(tmp).unlink(missing_ok=True)
                    print(f'[cache WARNING] Drive entry {drive.name} not '
                          f'usable: {type(e).__name__}: {e}')
                else:
                    self._say(f'[cache] hit  {label}  (restored Drive -> local '
                              f'{local.name})')
                    return entry['payload']

        self._say(f'[cache] miss {label} — building (this is the slow path)')
        payload = build()
        self.put(name, key, payload)
        return payload


# ---------------------------------------------------------------------------
# Cache keys
#
# Public so that a live session can bank an artifact it already computed under
# exactly the key the notebook will later look up.  Never inline these rules at
# a call site — a key that drifts is a cache that silently never hits.
# ---------------------------------------------------------------------------

def expressions_cache_key(
    pkl_path, tokenizer, *, max_seq_len: int, max_vars: int,
    max_expressions: int, dedupe_by_tokens: bool = True,
    max_per_sequence: int = 1,
) -> str:
    """Key for a deduplicated `load_synthetic_pkl` result.

    Covers the source pickle's identity and size, every argument that changes
    which expressions survive, the tokenizer vocabulary, and the source of the
    modules that do the parsing.
    """
    src = Path(pkl_path)
    return _hash_parts(
        src.name, src.stat().st_size,
        max_expressions, max_seq_len, max_vars,
        dedupe_by_tokens, max_per_sequence,
        tuple(tokenizer.vocab),
        module_fingerprint(_tokenizer_mod, _expressions_mod),
    )


def prefix_tree_cache_key(token_keys) -> str:
    """Key for a branching prefix tree, from the sequences that produce it."""
    token_keys = list(token_keys)
    return _hash_parts(
        _hash_token_keys(token_keys), len(token_keys),
        module_fingerprint(_evaluation_mod, _dataset_mod),
    )


# ---------------------------------------------------------------------------
# The two cached artifacts
# ---------------------------------------------------------------------------

def cached_synthetic_expressions(
    pkl_path, tokenizer, *, max_seq_len: int, max_vars: int,
    max_expressions: int, dedupe_by_tokens: bool = True,
    max_per_sequence: int = 1, cache: DataCache | None = None,
    progress: bool = True,
) -> list:
    """`load_synthetic_pkl` with a content-addressed cache."""
    src = Path(pkl_path)
    key = expressions_cache_key(
        src, tokenizer, max_seq_len=max_seq_len, max_vars=max_vars,
        max_expressions=max_expressions, dedupe_by_tokens=dedupe_by_tokens,
        max_per_sequence=max_per_sequence,
    )

    def _build():
        return load_synthetic_pkl(
            str(src), max_seq_len=max_seq_len, tokenizer=tokenizer,
            max_expressions=max_expressions,
            dedupe_by_tokens=dedupe_by_tokens,
            max_per_sequence=max_per_sequence,
            progress=progress,
        )

    if cache is None:
        return _build()
    exprs = cache.get_or_build(
        'expressions', key, _build,
        label=f'{max_expressions} raw expressions -> deduped')
    print(f'  {len(exprs)} deduplicated expressions '
          f'({max_expressions} raw strings considered)')
    return exprs


def cached_prefix_tree(token_keys, *, cache: DataCache | None = None,
                       progress: bool = True) -> dict:
    """`build_prefix_tree` with a content-addressed cache.

    Keyed on the training token sequences themselves, so the cached tree is
    provably the tree those sequences produce — there is no configuration to
    keep in sync, and a different split can never hit this entry.
    """
    token_keys = list(token_keys)
    key = prefix_tree_cache_key(token_keys)

    def _build():
        return build_prefix_tree(token_keys, progress=progress)

    if cache is None:
        return _build()
    return cache.get_or_build(
        'branch_tree', key, _build,
        label=f'branch tree over {len(token_keys)} train sequences')
