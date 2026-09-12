"""Dataset construction for the physics-blind baseline.

Everything physics-neutral is imported from the parent package so the baseline
sees exactly the same lines, the same seeded stratified split, the same target
tokenisation and the same vocabulary as the physics-informed model. Only the
graph featurisation differs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import random

import torch

from ..data_loading import load_lines_from_dir
from ..dataset import GraphExprDataset
from ..diagram_parser import parse_diagram
from ..expr_simplify import build_target_tokens
from ..pair_rules import PairRules
from ..tokenizer import Vocab
from ..train import MODEL_PARTICLES, _safe_encode, _stratified_split
from .graph_blind import build_blind_base_graph, build_blind_pair_graphs

# The blind baseline has no leg-permutation augmentation, so each example holds
# exactly one graph slot.
MAX_PERMS = 1


def build_blind_datasets(
    model_names,
    data_root: str,
    n_initial: int = 2,
    max_seq_len_cap: Optional[int] = None,
    seed: int = 42,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    orders: Optional[List[str]] = None,
    ranks: Optional[List[str]] = None,
    mass_rewrite: bool = False,
    mass_rewrite_mode: str = "leg_sets",
    canonicalize_targets: bool = False,
    compact_header_stub: bool = False,
    simplify_targets: bool = True,
    subsample: Optional[int] = None,
):
    if isinstance(model_names, str):
        model_names = [model_names]
    model_names = list(model_names)
    for m in model_names:
        if m not in MODEL_PARTICLES:
            raise ValueError(f"Unknown physics model {m!r}; expected one of {list(MODEL_PARTICLES)}")

    lines: List[Tuple[Any, str, Dict[str, object]]] = []
    for m in model_names:
        lines.extend(
            load_lines_from_dir(
                f"{data_root.rstrip('/')}/{m}",
                orders=orders,
                ranks=ranks,
                compact_header_stub=compact_header_stub,
            )
        )
    if not lines:
        raise RuntimeError(
            f"No dataset lines loaded from {data_root!r} for models={model_names}, "
            f"orders={orders}, ranks={ranks}."
        )

    if subsample is not None and 0 < subsample < len(lines):
        lines = random.Random(seed).sample(lines, subsample)
        print(f"[data] subsampled {subsample} of the loaded lines (seed={seed})")

    train_lines, val_lines, test_lines = _stratified_split(
        lines, seed=seed, train_frac=train_frac, val_frac=val_frac
    )

    strata = Counter((m["model"], m["order"], m["rank"]) for _, _, m in lines)
    if strata:
        print("[data] strata (lines per model/order/rank):")
        for key in sorted(strata.keys()):
            print(f"  {key}: {strata[key]}")

    def _prepare(diagrams, expr: str, meta: Dict[str, Any]):
        if isinstance(diagrams, str):
            diagrams = (diagrams, None)
        toks, slot_map = build_target_tokens(
            diagrams,
            expr,
            meta,
            mass_rewrite_mode=mass_rewrite_mode,
            mass_rewrite=mass_rewrite,
            simplify_targets=simplify_targets,
            canonicalize_targets=canonicalize_targets,
        )
        diag_i, diag_j = diagrams
        bd_i = parse_diagram(diag_i)
        bd_j = parse_diagram(diag_j) if (diag_j is not None and diag_j != diag_i) else None
        meta_out = dict(meta)
        meta_out["slot_map"] = slot_map
        return (bd_i, bd_j), toks, meta_out

    def _graphs_for(bds, meta):
        bd_i, bd_j = bds
        rules = PairRules(enforce_particle_match=True, enforce_bijection=True,
                          group_by_particle=False)
        particles = MODEL_PARTICLES.get(meta.get("model"), [])
        built_i = build_blind_base_graph(bd_i, particles=particles, n_initial=n_initial)
        built_j = (
            build_blind_base_graph(bd_j, particles=particles, n_initial=n_initial)
            if bd_j is not None else None
        )
        pair_graphs = build_blind_pair_graphs(built_i, built_j, rules=rules)
        graphs = [pg.data for pg in pair_graphs][:MAX_PERMS]
        if len(graphs) < MAX_PERMS:
            graphs += [None] * (MAX_PERMS - len(graphs))
        return graphs

    train_cache: List[Tuple[List, List[str], Dict[str, Any]]] = []
    train_tok_for_vocab: List[List[str]] = []
    skipped_train, long_filtered_train = 0, 0
    n_train_lines = len(train_lines)
    for i, (diagrams, expr, meta) in enumerate(train_lines):
        if simplify_targets and i > 0 and i % 200 == 0:
            print(f"[data] sympy simplify: {i}/{n_train_lines} train lines...", flush=True)
        try:
            bds, toks, meta_full = _prepare(diagrams, expr, meta)
            if max_seq_len_cap is not None and 1 + len(toks) > max_seq_len_cap:
                long_filtered_train += 1
                continue
            graphs = _graphs_for(bds, meta)
            if not graphs:
                skipped_train += 1
                continue
            train_cache.append((graphs, toks, meta_full))
            train_tok_for_vocab.append(toks)
        except Exception:
            skipped_train += 1
            continue

    if not train_tok_for_vocab:
        raise RuntimeError("No usable training examples after parsing/filtering; cannot build vocab.")

    vocab = Vocab.build(train_tok_for_vocab, min_freq=1)

    def _encode(graphs, toks, meta):
        ids = _safe_encode(vocab, toks)
        y = torch.tensor([vocab.bos] + ids + [vocab.eos], dtype=torch.long)
        return (graphs, y[:-1], y[1:], True, meta)

    def _build_split_items(split_lines, *, use_cache=None):
        items = []
        skipped, long_filtered, ood = 0, 0, 0

        if use_cache is not None:
            for graphs, toks, meta in use_cache:
                if any(t not in vocab.stoi for t in toks):
                    ood += 1
                    continue
                items.append(_encode(graphs, toks, meta))
            return GraphExprDataset(items), skipped, long_filtered, ood

        for diagrams, expr, meta in split_lines:
            try:
                bds, toks, meta_full = _prepare(diagrams, expr, meta)
                if max_seq_len_cap is not None and 1 + len(toks) > max_seq_len_cap:
                    long_filtered += 1
                    continue
                has_unk = any(t not in vocab.stoi for t in toks)
                if has_unk:
                    ood += 1
                graphs = _graphs_for(bds, meta)
                if not graphs:
                    skipped += 1
                    continue
                item = _encode(graphs, toks, meta_full)
                # Targets with out-of-vocabulary tokens are excluded from loss and metrics,
                # matching the parent pipeline.
                items.append((item[0], item[1], item[2], not has_unk, item[4]))
            except Exception:
                skipped += 1
                continue
        return GraphExprDataset(items), skipped, long_filtered, ood

    train_ds, _, _, ood_train = _build_split_items([], use_cache=train_cache)
    val_ds, skipped_val, long_val, ood_val = _build_split_items(val_lines)
    test_ds, skipped_test, long_test, ood_test = _build_split_items(test_lines)

    print(
        f"Loaded {len(lines)} lines. "
        f"Train/val/test lines: {len(train_lines)}/{len(val_lines)}/{len(test_lines)}. "
        f"Kept examples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}. "
        f"Skipped: {skipped_train}/{skipped_val}/{skipped_test}. "
        f"Filtered by cap: {long_filtered_train}/{long_val}/{long_test}. "
        f"OOD targets (contain OOV tokens vs train-vocab): {ood_train}/{ood_val}/{ood_test}."
    )
    return train_ds, val_ds, test_ds, vocab
