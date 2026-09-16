#!/usr/bin/env python
"""Build the canonical-template dataset from the raw synthetic pickle.

Deduplicating the raw corpus is the slow step in the pipeline — every string
goes through SymPy at ~3 ms, so 200 000 of them cost 10-15 minutes, paid again
on every fresh Colab runtime.  It is also a pure function of the corpus and the
tokenizer vocabulary, so it does not belong in a training run at all.  This
script does it once and writes the canonical forms to a new file.

What it emits, per canonical form:

* the coefficient **template** (``c0*sin(c1*x1) + c2``), whose prefix is the
  decoder target;
* the **reference realisation** — the original SymPy expression, kept verbatim
  so ``dynamic_constants=False`` reproduces the pre-template pipeline exactly;
* per-slot **constant pools** harvested from every raw string that deduplicated
  away.  ~200 000 strings collapse to ~1 500 forms, so instead of discarding
  the other ~198 500 realisations the sampler gets ~130 observed values per
  coefficient slot to draw from.

The source pickle is only read.  The output is a new file; nothing existing is
modified or overwritten unless ``--output`` names it and ``--force`` is passed.

Example::

    python tools/build_template_dataset.py \
        --input data/synthetic.pkl \
        --output data/synthetic_templates.pkl \
        --max-expressions 200000

``--max-expressions 200000`` matches ``MAX_SYNTH`` in ``jepa_pretrain_v2*``, so
the canonical forms — and therefore the train/val/test split at a given seed —
are the same ones those experiments use.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from symbolic_jepa.templates import (  # noqa: E402
    build_templates_from_strings, save_template_dataset,
)
from symbolic_jepa.tokenizer import PrefixTokenizer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data/synthetic.pkl',
                   help='Raw expression-string pickle (read only).')
    p.add_argument('--output', default='data/synthetic_templates.pkl',
                   help='Destination for the canonical templates.')
    p.add_argument('--max-expressions', type=int, default=200_000,
                   help='Raw strings to read; 0 reads all. Default matches '
                        'MAX_SYNTH in jepa_pretrain_v2.')
    p.add_argument('--max-seq-len', type=int, default=64)
    p.add_argument('--max-vars', type=int, default=1,
                   help='Tokenizer variable count (synthetic data is univariate).')
    p.add_argument('--pool-cap', type=int, default=512,
                   help='Max observed values retained per coefficient slot.')
    p.add_argument('--pool-expressions', type=int, default=0,
                   help='Read this many strings to fill the coefficient pools '
                        'while still taking the canonical form set from the '
                        'first --max-expressions. Deepens the pools without '
                        'changing the forms or the split, at the cost of a '
                        'longer parse. 0 (default) reads --max-expressions.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true',
                   help='Overwrite --output if it already exists.')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f'error: input {src} does not exist', file=sys.stderr)
        return 2
    if dst.exists() and not args.force:
        print(f'error: {dst} already exists; pass --force to overwrite',
              file=sys.stderr)
        return 2

    tokenizer = PrefixTokenizer(max_vars=args.max_vars)

    print(f'Reading {src} ...')
    with open(src, 'rb') as f:
        strings = pickle.load(f)
    print(f'  {len(strings)} raw expression strings')

    n_forms_from = args.max_expressions or len(strings)
    n_read = max(n_forms_from, args.pool_expressions) or len(strings)
    strings = strings[:n_read]
    form_horizon = n_forms_from if args.pool_expressions > n_forms_from else 0
    print(f'  reading {len(strings)}'
          + (f'; canonical forms taken from the first {n_forms_from}'
             if form_horizon else ''))

    t0 = time.time()
    templates, stats = build_templates_from_strings(
        strings, tokenizer,
        max_seq_len=args.max_seq_len,
        pool_cap=args.pool_cap,
        form_horizon=form_horizon,
        seed=args.seed,
        progress=not args.quiet,
    )
    elapsed = time.time() - t0

    print(f'\nBuilt {len(templates)} canonical forms in {elapsed / 60:.1f} min')
    print(f'  parse failures      : {stats["n_parse_failed"]}')
    print(f'  tokenize failures   : {stats["n_tokenize_failed"]}')
    print(f'  too long (> {args.max_seq_len})   : {stats["n_too_long"]}')
    print(f'  unknown tokens      : {stats["n_unk"]}')
    print(f'  prefix mismatches   : {stats["n_prefix_mismatch"]}')
    print(f'  slot mismatches     : {stats["n_slot_mismatch"]}')
    print(f'  realisations pooled : {stats["n_pooled"]}')
    if form_horizon:
        print(f'  new forms suppressed beyond horizon: '
              f'{stats["n_beyond_horizon"]}')
    print(f'  pool size per slot  : min {stats["pool_min"]}, '
          f'median {stats["pool_median"]:.0f}, max {stats["pool_max"]}')
    print(f'  forms with a single realisation: {stats["n_single_realization"]}')

    slots = np.array([t.n_constants for t in templates])
    if slots.size:
        print(f'  coefficient slots   : min {slots.min()}, '
              f'median {np.median(slots):.0f}, max {slots.max()}')

    meta = dict(stats)
    meta.update({
        'source_pkl': src.name,
        'source_size': src.stat().st_size,
        'max_expressions': args.max_expressions,
        'max_seq_len': args.max_seq_len,
        'max_vars': args.max_vars,
        'pool_cap': args.pool_cap,
        'seed': args.seed,
        'build_seconds': round(elapsed, 1),
    })

    save_template_dataset(dst, templates, tokenizer, meta=meta)
    print(f'\nWrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
