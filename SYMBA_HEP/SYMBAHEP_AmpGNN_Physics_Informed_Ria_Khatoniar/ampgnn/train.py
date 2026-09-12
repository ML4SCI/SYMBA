import argparse, glob, hashlib, json, os, random, re, sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader.dataloader import Collater
from torch.nn.utils.rnn import pad_sequence

from .data_loading import load_lines_from_dir
from .diagram_parser import parse_diagram, BaseDiagram
from .graph_builder import (
    EDGE_FEAT_DIM,
    GRAPH_FEAT_DIM,
    build_base_graph,
    build_pair_graphs,
)
from .mass_rewrite import canonicalize_masses
from .expr_canonical import canonicalize_tokens
from .expr_simplify import build_target_tokens, coupling_for_model
from .pair_rules import PairRules
from .tokenizer import tokenize_expr, Vocab
from .dataset import GraphExprDataset
from .model import LitAmpGNN
from .term_slots import (
    TermSlotCacheMissError,
    TermSlotVocabulary,
    build_term_slot_target,
    slot_vocabulary_hash,
    term_slot_cache_stats,
    validate_term_slot_target,
)
from .term_slot_objective import collate_term_slot_targets

MODEL_PARTICLES = {
    "QED": ["e", "mu", "t", "u", "d", "s", "tt", "c", "b", "A"],
    "QCD": ["u", "d", "s", "t", "c", "b", "G"],
    "EW":  ["u_R","u_L","d_L","d_R","e_R","e_L","nue_L","W","h","A","Z","e","u","d"],
}


def _safe_encode(vocab: Vocab, toks: List[str]) -> List[int]:
    try:
        return vocab.encode(toks)
    except Exception:
        unk_id = getattr(vocab, "unk", None)
        pad_id = getattr(vocab, "pad", 0)
        ids: List[int] = []
        for t in toks:
            if t in vocab.stoi:
                ids.append(int(vocab.stoi[t]))
            else:
                ids.append(int(unk_id) if unk_id is not None else int(pad_id))
        return ids


def _stratified_split(
    lines: List[Tuple[str, str, Dict[str, str]]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List, List, List]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            "train_frac + val_frac must be less than 1 so the test split is non-empty"
        )

    buckets: Dict[Tuple[str, str, str], List[int]] = {}
    for i, (_, _, meta) in enumerate(lines):
        key = (meta["model"], meta["order"], meta["rank"])
        buckets.setdefault(key, []).append(i)

    g = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for key in sorted(buckets.keys()):
        idxs = buckets[key]
        perm = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[p] for p in perm]
        n = len(shuffled)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
    return (
        [lines[i] for i in train],
        [lines[i] for i in val],
        [lines[i] for i in test],
    )


def build_datasets(
    model_names,
    data_root: str,
    n_initial: int = 2,
    max_seq_len_cap: int = None,
    max_perms: int = 12,
    seed: int = 42,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    orders: Optional[List[str]] = None,
    ranks: Optional[List[str]] = None,
    mass_rewrite: bool = True,
    mass_rewrite_mode: str = "species",
    canonicalize_targets: bool = True,
    compact_header_stub: bool = False,
    simplify_targets: bool = False,
    subsample: Optional[int] = None,
    split_seed: Optional[int] = None,
    output_mode: str = "sequence",
    slot_max_exponent: int = 8,
    slot_num_slots: int = 64,
    slot_num_factors: int = 3,
    slot_den_slots: int = 4,
    slot_den_terms: int = 5,
    slot_den_term_factors: int = 3,
):
    if isinstance(model_names, str):
        model_names = [model_names]
    model_names = list(model_names)
    for m in model_names:
        if m not in MODEL_PARTICLES:
            raise ValueError(f"Unknown physics model {m!r}; expected one of {list(MODEL_PARTICLES)}")

    lines: List[Tuple[str, str, Dict[str, object]]] = []
    for m in model_names:
        folder = os.path.join(data_root, m)
        lines.extend(
            load_lines_from_dir(
                folder,
                orders=orders,
                ranks=ranks,
                compact_header_stub=compact_header_stub,
            )
        )

    if not lines:
        raise RuntimeError(
            f"No dataset lines loaded from {data_root!r} for models={model_names}, "
            f"orders={orders}, ranks={ranks}. "
            "Supported formats: legacy single-diagram rows with `Vertex ...`, "
            "7-field pair rows (header : pair i-j : diagram_i : diagram_j : "
            "M_i : M_j : term), or compact 5-field pair rows "
            "(header : pair i-j : M_i : M_j : term) with `Diagram N: Vertex ...` "
            "sidecars or --compact_header_stub."
        )

    if subsample is not None and subsample > 0 and subsample < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, subsample)
        print(f"[data] subsampled {subsample} of the loaded lines (seed={seed})")

    if output_mode not in ("sequence", "term_slots"):
        raise ValueError(f"unsupported output_mode {output_mode!r}")
    effective_split_seed = int(seed if split_seed is None else split_seed)
    train_lines, val_lines, test_lines = _stratified_split(
        lines, seed=effective_split_seed, train_frac=train_frac, val_frac=val_frac
    )
    print(
        f"[data] split (seed={effective_split_seed}): "
        f"train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)} "
        f"(fractions={train_frac:g}/{val_frac:g}/{1.0 - train_frac - val_frac:g})"
    )

    strata = Counter((m["model"], m["order"], m["rank"]) for _, _, m in lines)
    if strata:
        print("[data] strata (lines per model/order/rank):")
        for key in sorted(strata.keys()):
            print(f"  {key}: {strata[key]}")

    def _precompute_anchor_dists_cpu(data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        N = x.size(0)

        is_external = x[:, 0] > 0.5
        side = x[:, -1] > 0.5
        amp_ext = is_external & (~side)
        conj_ext = is_external & side


        cross_mask = edge_attr[:, 2] > 0.5
        ei = edge_index[:, ~cross_mask]


        adj = [[] for _ in range(N)]
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            adj[u].append(v)


        partner = [-1] * N
        ce = edge_index[:, cross_mask]
        src_c = ce[0].tolist()
        dst_c = ce[1].tolist()
        amp_ext_list = amp_ext.tolist()
        conj_ext_list = conj_ext.tolist()
        for u, v in zip(src_c, dst_c):
            if amp_ext_list[u] and conj_ext_list[v]:
                partner[u] = v

        amp_nodes = torch.nonzero(amp_ext, as_tuple=False).view(-1).tolist()
        K = len(amp_nodes)


        denom = max(N - 1, 1)

        import collections
        out = torch.empty((N, K), dtype=torch.float32)
        for i, a in enumerate(amp_nodes):
            sources = [a]
            b = partner[a]
            if b != -1:
                sources.append(b)

            dist = [-1] * N
            q = collections.deque()
            for s in sources:
                dist[s] = 0
                q.append(s)

            while q:
                u = q.popleft()
                du = dist[u]
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = du + 1
                        q.append(w)


            col = [1.0 if d < 0 else float(min(d, denom)) / float(denom) for d in dist]
            out[:, i] = torch.tensor(col, dtype=torch.float32)

        data.anchor_dists = out
        return data

    def _pack_graphs(pair_graphs):
        graphs = [pg.data for pg in pair_graphs][:max_perms]
        if len(graphs) < max_perms:
            graphs += [None] * (max_perms - len(graphs))
        return graphs

    def _bd_for_masses(bd_i, bd_j):
        if bd_j is None:
            return bd_i
        merged_off = dict(bd_i.offshell_by_type)
        for sp, vids in bd_j.offshell_by_type.items():
            merged_off.setdefault(sp, vids)
        return BaseDiagram(vertices=bd_i.vertices, externals=bd_i.externals,
                           offshell_by_type=merged_off, internal_edges=bd_i.internal_edges)

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
        if output_mode == "term_slots":
            slot_target = build_term_slot_target(
                toks, coupling_for_model(str(meta.get("model", "QCD")))
            )
            validate_term_slot_target(
                slot_target,
                max_num_terms=slot_num_slots,
                max_num_factors=slot_num_factors,
                max_den_factors=slot_den_slots,
                max_den_terms=slot_den_terms,
                max_den_term_factors=slot_den_term_factors,
                max_exponent=slot_max_exponent,
            )
            degree_bd_j = bd_i if bd_j is None else bd_j
            if len(bd_i.externals) != len(degree_bd_j.externals):
                raise ValueError("diagram pair has inconsistent external-leg counts")
            denominator_degree = len(bd_i.internal_edges) + len(
                degree_bd_j.internal_edges
            )
            numerator_degree = denominator_degree + 4 - len(bd_i.externals)
            if numerator_degree < 0:
                raise ValueError("topology implies a negative numerator degree")
            slot_target["topology_degree"] = {
                "numerator_degree2": 2 * numerator_degree,
                "denominator_degree": denominator_degree,
            }
            meta_out["term_slot_target_raw"] = slot_target
        return (bd_i, bd_j), toks, meta_out

    def _build_pair_graphs_for(bds, meta):
        bd_i, bd_j = bds
        rules = PairRules(enforce_particle_match=True, enforce_bijection=True, group_by_particle=False)
        sample_model = meta.get("model")
        particles = MODEL_PARTICLES.get(sample_model, [])
        built_i = build_base_graph(bd_i, particles=particles, n_initial=n_initial, model=sample_model)
        built_j = (
            build_base_graph(bd_j, particles=particles, n_initial=n_initial, model=sample_model)
            if bd_j is not None else None
        )
        return build_pair_graphs(built_i, built_j, limit_per_group=999, rules=rules)


    train_cache = []
    train_tok_for_vocab: List[List[str]] = []
    train_slot_targets: List[Dict[str, object]] = []
    skipped_train, long_filtered_train = 0, 0
    n_train_lines = len(train_lines)
    for i, (diagrams, expr, meta) in enumerate(train_lines):
        if simplify_targets and i > 0 and i % 200 == 0:
            print(f"[data] sympy simplify: {i}/{n_train_lines} train lines...", flush=True)
        try:
            bds, toks, meta_full = _prepare(diagrams, expr, meta)
            seq_len = 1 + len(toks)
            if max_seq_len_cap is not None and seq_len > max_seq_len_cap:
                long_filtered_train += 1
                continue

            pair_graphs = _build_pair_graphs_for(bds, meta)
            if not pair_graphs:
                skipped_train += 1
                continue
            for pg in pair_graphs:
                pg.data = _precompute_anchor_dists_cpu(pg.data)
            graphs = _pack_graphs(pair_graphs)
            train_cache.append((graphs, toks, meta_full))
            train_tok_for_vocab.append(toks)
            if output_mode == "term_slots":
                train_slot_targets.append(meta_full["term_slot_target_raw"])
        except TermSlotCacheMissError:
            raise
        except Exception:
            skipped_train += 1
            continue

    if not train_tok_for_vocab:
        raise RuntimeError("No usable training examples after parsing/filtering; cannot build vocab.")


    vocab = Vocab.build(train_tok_for_vocab, min_freq=1)
    slot_vocab: Optional[TermSlotVocabulary] = None
    if output_mode == "term_slots":
        slot_vocab = TermSlotVocabulary.build(train_slot_targets)
        setattr(vocab, "term_slot_vocabulary", slot_vocab)
        print(
            "[data] term-slot vocab: "
            f"coefficients={slot_vocab.coefficient_size}, "
            f"symbols={slot_vocab.symbol_size}, "
            f"couplings={slot_vocab.coupling_size}, "
            f"hash={slot_vocabulary_hash(slot_vocab)}"
        )

    def _build_split_items(split_lines, *, use_cache=None, mark_ood_loss: bool = True):
        items = []
        skipped, long_filtered, ood = 0, 0, 0

        if use_cache is not None:
            for graphs, toks, meta in use_cache:
                has_unk = any(t not in vocab.stoi for t in toks)
                encoded_slot_target = None
                if slot_vocab is not None:
                    encoded_slot_target, slot_has_unk = slot_vocab.encode(
                        meta["term_slot_target_raw"]
                    )
                    has_unk = has_unk or slot_has_unk
                keep_loss = (not has_unk) if mark_ood_loss else True
                if has_unk:
                    ood += 1
                    continue
                ids = _safe_encode(vocab, toks)
                y = torch.tensor([vocab.bos] + ids + [vocab.eos], dtype=torch.long)
                meta_item = dict(meta)
                if encoded_slot_target is not None:
                    meta_item["term_slot_target"] = encoded_slot_target
                items.append((graphs, y[:-1], y[1:], keep_loss, meta_item))
            return GraphExprDataset(items), skipped, long_filtered, ood

        for diagrams, expr, meta in split_lines:
            try:
                bds, toks, meta_full = _prepare(diagrams, expr, meta)
                seq_len = 1 + len(toks)
                if max_seq_len_cap is not None and seq_len > max_seq_len_cap:
                    long_filtered += 1
                    continue

                has_unk = any(t not in vocab.stoi for t in toks)
                encoded_slot_target = None
                if slot_vocab is not None:
                    encoded_slot_target, slot_has_unk = slot_vocab.encode(
                        meta_full["term_slot_target_raw"]
                    )
                    has_unk = has_unk or slot_has_unk
                keep_loss = (not has_unk) if mark_ood_loss else True
                if has_unk:
                    ood += 1

                pair_graphs = _build_pair_graphs_for(bds, meta)
                if not pair_graphs:
                    skipped += 1
                    continue

                for pg in pair_graphs:
                    pg.data = _precompute_anchor_dists_cpu(pg.data)
                graphs = _pack_graphs(pair_graphs)
                ids = _safe_encode(vocab, toks)
                y = torch.tensor([vocab.bos] + ids + [vocab.eos], dtype=torch.long)
                if encoded_slot_target is not None:
                    meta_full["term_slot_target"] = encoded_slot_target
                items.append((graphs, y[:-1], y[1:], keep_loss, meta_full))
            except TermSlotCacheMissError:
                raise
            except Exception:
                skipped += 1
                continue
        return GraphExprDataset(items), skipped, long_filtered, ood

    train_ds, _, _, ood_train = _build_split_items([], use_cache=train_cache, mark_ood_loss=True)
    val_ds, skipped_val, long_val, ood_val = _build_split_items(val_lines, mark_ood_loss=True)
    test_ds, skipped_test, long_test, ood_test = _build_split_items(test_lines, mark_ood_loss=True)

    print(
        f"Loaded {len(lines)} lines. "
        f"Train/val/test lines: {len(train_lines)}/{len(val_lines)}/{len(test_lines)}. "
        f"Kept examples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}. "
        f"Skipped: {skipped_train}/{skipped_val}/{skipped_test}. "
        f"Filtered by cap: {long_filtered_train}/{long_val}/{long_test}. "
        f"OOD targets (contain OOV tokens vs train-vocab): {ood_train}/{ood_val}/{ood_test}."
    )
    if output_mode == "term_slots":
        cache_stats = term_slot_cache_stats()
        if cache_stats["path"]:
            print(
                "[data] term-slot SQL cache: "
                f"hits={cache_stats['hits']} misses={cache_stats['misses']} "
                f"readonly={cache_stats['readonly']} path={cache_stats['path']}"
            )


    rank_anchor_dims: Dict[str, int] = {}
    for ds in (train_ds, val_ds, test_ds):
        for item in ds.items:
            graphs, _, _, _, meta = item
            sample = next((g for g in graphs if g is not None), None)
            if sample is None or getattr(sample, "anchor_dists", None) is None:
                continue
            K = int(sample.anchor_dists.size(1))
            rank_anchor_dims.setdefault(meta["rank"], K)
    print(f"[data] rank -> anchor count: {rank_anchor_dims}")

    return train_ds, val_ds, test_ds, vocab, rank_anchor_dims


def _process_key(meta: Dict[str, Any]) -> str:
    return f"{meta.get('model', 'UNK')}_{meta.get('rank', 'unk')}"


class RankBucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        buckets: Dict[str, List[int]] = {}
        for i, item in enumerate(dataset.items):
            meta = item[4]
            buckets.setdefault(_process_key(meta), []).append(i)
        self.buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        all_batches: List[List[int]] = []
        for key in sorted(self.buckets.keys()):
            idxs = list(self.buckets[key])
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[p] for p in perm]
            for s in range(0, len(idxs), self.batch_size):
                batch = idxs[s:s + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in order]
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.buckets.values():
            n = len(idxs)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def make_collate_grouped(
    max_perms: int,
    pad_id: int,
    term_slot_layout: Optional[Dict[str, int]] = None,
):
    collater = Collater([], exclude_keys=[])
    def collate(batch):
        graphs_grouped, y_in_list, y_out_list, keep_list, meta_list = zip(*batch)
        B = len(graphs_grouped)

        masks = torch.zeros(B, max_perms, dtype=torch.bool)
        batches_per_p, idx_per_p = [], []
        for p in range(max_perms):
            gs, idxs = [], []
            for b, graphs in enumerate(graphs_grouped):
                g = graphs[p]
                if g is not None:
                    gs.append(g); idxs.append(b); masks[b, p] = True
            batches_per_p.append(collater(gs) if gs else None)
            idx_per_p.append(torch.as_tensor(idxs, dtype=torch.long))

        y_in  = pad_sequence(y_in_list,  batch_first=True, padding_value=pad_id)
        y_out = pad_sequence(y_out_list, batch_first=True, padding_value=pad_id)
        keep_loss = torch.as_tensor(keep_list, dtype=torch.bool)

        result = (
            batches_per_p, idx_per_p, masks, y_in, y_out, keep_loss, list(meta_list)
        )
        if term_slot_layout is None:
            return result
        if any("term_slot_target" not in meta for meta in meta_list):
            raise RuntimeError("term-slot batch is missing structured targets")
        dense_targets = collate_term_slot_targets(
            [meta["term_slot_target"] for meta in meta_list],
            keep_loss=keep_list,
            **term_slot_layout,
        )
        return result + (dense_targets,)
    return collate


class GraphExprDataModule(pl.LightningDataModule):
    def __init__(self, dataset_or_splits, batch_size: int=8, num_workers: int=0,
                 seed: int=42, max_perms: int=12, pad_id: int=0,
                 term_slot_layout: Optional[Dict[str, int]] = None):
        super().__init__()
        if isinstance(dataset_or_splits, (tuple, list)) and len(dataset_or_splits) == 3:
            self.dataset = None
            self.train_set, self.val_set, self.test_set = dataset_or_splits
        else:
            self.dataset = dataset_or_splits
            self.train_set = None
            self.val_set = None
            self.test_set = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.max_perms = max_perms
        self.pad_id = pad_id
        self.term_slot_layout = term_slot_layout

    def setup(self, stage=None):
        if self.train_set is not None:
            return
        if self.dataset is None:
            return
        n = len(self.dataset)
        n_train = int(0.90 * n)
        n_val = int(0.05 * n)
        n_test = n - n_train - n_val
        g = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, [n_train, n_val, n_test], generator=g)

    def _loader(self, split, shuffle: bool) -> DataLoader:
        batch_size = self.batch_size if shuffle else 1
        if hasattr(split, "items"):
            sampler = RankBucketBatchSampler(
                split, batch_size=batch_size, shuffle=shuffle, seed=self.seed
            )
            return DataLoader(
                split,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=make_collate_grouped(
                    self.max_perms,
                    pad_id=self.pad_id,
                    term_slot_layout=self.term_slot_layout,
                ),
            )
        return DataLoader(
            split, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=make_collate_grouped(
                self.max_perms,
                pad_id=self.pad_id,
                term_slot_layout=self.term_slot_layout,
            ),
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)

RUN_STATE_FILENAME = "run_state.json"


def _hash_dict(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_run_state(args: argparse.Namespace, vocab: Vocab,
                    in_dim: int, edge_dim: int, graph_feat_dim: int, num_anchors: int,
                    rank_anchor_dims: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    return {
        "model": args.model,
        "data_dir": os.path.abspath(args.data_dir),
        "orders": args.orders,
        "ranks": args.ranks,
        "split_seed": int(getattr(args, "split_seed", args.seed)),
        "train_frac": float(getattr(args, "train_frac", 0.90)),
        "val_frac": float(getattr(args, "val_frac", 0.05)),
        "n_initial": args.n_initial,
        "max_seq_len_cap": args.max_seq_len_cap,
        "mass_rewrite": bool(args.mass_rewrite),
        "mass_rewrite_mode": str(args.mass_rewrite_mode),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "accumulate_grad_batches": int(getattr(args, "accumulate_grad_batches", 1)),
        "scheduler": str(args.scheduler),
        "warmup_steps": int(args.warmup_steps),
        "phys_decode": str(getattr(args, "phys_decode", "none")),
        "canonicalize_targets": bool(getattr(args, "canonicalize_targets", True)),
        "simplify_targets": bool(getattr(args, "simplify_targets", True)),
        "subsample": getattr(args, "subsample", None),
        "canonicalize_eval": bool(getattr(args, "canonicalize_eval", True)),
        "vocab_size": len(vocab.stoi),
        "vocab_hash": _hash_dict(vocab.stoi),
        "in_dim": int(in_dim),
        "edge_dim": int(edge_dim),
        "graph_feat_dim": int(graph_feat_dim),
        "num_anchors": int(num_anchors),
        "rank_anchor_dims": dict(rank_anchor_dims) if rank_anchor_dims else {},
        "loss_mode": str(args.loss_mode),
        "output_mode": str(getattr(args, "output_mode", "sequence")),
        "max_perms": int(args.max_perms),
        "enc_hid": int(args.enc_hid),
        "enc_layers": args.enc_layers,
        "enc_heads": args.enc_heads,
        "enc_dropout": args.enc_dropout,
        "d_model": args.d_model,
        "dec_nhead": args.dec_nhead,
        "dec_layers": args.dec_layers,
        "dec_dropout": args.dec_dropout,
        "dec_max_len": args.dec_max_len,
        "slot_max_exponent": int(getattr(args, "slot_max_exponent", 8)),
        "slot_num_slots": int(getattr(args, "slot_num_slots", 64)),
        "slot_num_factors": int(getattr(args, "slot_num_factors", 3)),
        "slot_den_slots": int(getattr(args, "slot_den_slots", 4)),
        "slot_den_terms": int(getattr(args, "slot_den_terms", 5)),
        "slot_den_term_factors": int(getattr(args, "slot_den_term_factors", 3)),
        "slot_degree_loss_weight": float(
            getattr(args, "slot_degree_loss_weight", 0.1)
        ),
        "slot_no_object_weight": float(
            getattr(args, "slot_no_object_weight", 0.1)
        ),
        "slot_assignment": (
            "canonical" if getattr(args, "output_mode", "sequence") == "term_slots"
            else None
        ),
        "slot_vocab_hash": (
            slot_vocabulary_hash(vocab.term_slot_vocabulary)
            if getattr(vocab, "term_slot_vocabulary", None) is not None
            else None
        ),
    }


def _ckpt_global_step(path: str) -> int:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return -1
    step = ck.get("global_step", None)
    if step is None:
        return -1
    try:
        return int(step)
    except Exception:
        return -1


def discover_resume_ckpt(out_dir: str) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    last_candidates = [
        p for p in glob.glob(os.path.join(out_dir, "last*.ckpt"))
        if os.path.isfile(p)
    ]
    if last_candidates:
        scored = [(p, _ckpt_global_step(p)) for p in last_candidates]
        scored.sort(key=lambda kv: (kv[1], os.path.getmtime(kv[0])), reverse=True)
        return scored[0][0]
    candidates = [p for p in glob.glob(os.path.join(out_dir, "*.ckpt")) if os.path.isfile(p)]
    if not candidates:
        return None
    scored = [(p, _ckpt_global_step(p)) for p in candidates]
    scored.sort(key=lambda kv: (kv[1], os.path.getmtime(kv[0])), reverse=True)
    return scored[0][0]


def check_run_state_compatible(out_dir: str, current: Dict[str, Any]) -> Tuple[bool, str]:
    path = os.path.join(out_dir, RUN_STATE_FILENAME)
    if not os.path.isfile(path):
        return False, (
            f"no run_state.json next to the discovered checkpoint in {out_dir}. "
            "This checkpoint was likely produced by an older version of the code "
            "whose model architecture or vocab differs from the current one, and "
            "blindly resuming would crash the load. Pass --resume_from PATH "
            "explicitly if you're sure the shapes match, or --no-auto_resume to "
            "start fresh."
        )
    try:
        with open(path) as f:
            prior = json.load(f)
    except Exception as e:
        return False, f"could not read {path}: {e}"

    critical = ("model", "data_dir", "orders", "ranks", "split_seed",
                "train_frac", "val_frac", "n_initial",
                "max_seq_len_cap", "mass_rewrite", "mass_rewrite_mode",
                "in_dim", "edge_dim", "graph_feat_dim", "num_anchors", "vocab_hash",
                "rank_anchor_dims",
                "epochs", "batch_size", "accumulate_grad_batches", "scheduler", "warmup_steps",
                "loss_mode", "output_mode", "max_perms", "enc_hid",
                "enc_layers", "enc_heads", "enc_dropout", "d_model",
                "dec_nhead", "dec_layers", "dec_dropout", "dec_max_len",
                "slot_max_exponent", "slot_num_slots", "slot_num_factors",
                "slot_den_slots", "slot_den_terms", "slot_den_term_factors",
                "slot_degree_loss_weight", "slot_no_object_weight",
                "slot_assignment", "slot_vocab_hash")
    diffs = []
    for k in critical:
        if prior.get(k) != current.get(k):
            diffs.append(f"    {k}: was {prior.get(k)!r}, now {current.get(k)!r}")
    if diffs:
        return False, "run_state mismatch:\n" + "\n".join(diffs)
    return True, "run_state matches"


def write_run_state(out_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RUN_STATE_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def write_result_json(out_dir: str, filename: str, payload: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


class CompactEpochPrinter(pl.Callback):
    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
        value = metrics.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs <= 0 or trainer.sanity_checking:
            return
        if getattr(pl_module, "global_rank", 0) != 0:
            return

        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics
        fields = [
            ("train_loss", "train/loss_epoch"),
            ("val_loss", "val/loss"),
            ("val_token_acc", "val/token_acc"),
            ("val_seq_acc", "val/seq_acc"),
        ]

        parts = [f"[epoch {epoch}/{trainer.max_epochs}]"]
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr")
            if lr is not None:
                parts.append(f"lr={float(lr):.3e}")
        for label, key in fields:
            value = self._metric_value(metrics, key)
            if value is not None:
                parts.append(f"{label}={value:.6f}")
        print(" ".join(parts), flush=True)


def _parse_model_list(raw: str) -> List[str]:
    s = raw.strip()
    if s.lower() == "all":
        return list(MODEL_PARTICLES.keys())
    out: List[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in MODEL_PARTICLES:
            raise ValueError(f"Unknown physics model {tok!r}; expected one of {list(MODEL_PARTICLES)} or 'all'")
        if tok not in out:
            out.append(tok)
    if not out:
        raise ValueError(f"--model produced an empty model list from {raw!r}")
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True,
                    help="Physics model to train on. One of 'QED', 'QCD', 'EW', a comma-separated "
                         "list like 'QED,QCD,EW' for joint training across physics models, or 'all' "
                         "for every model present under --data_dir.")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n_initial", type=int, default=2)
    ap.add_argument("--orders", default="tree",
                    help="Comma-separated list of perturbative orders to include "
                         "(e.g. 'tree', 'tree,1_loop'). Pass 'all' to include every order present.")
    ap.add_argument("--ranks", default=None,
                    help="Comma-separated list of ranks to include (e.g. '2_to_2,2_to_3'). "
                         "Default = all ranks found under the selected orders.")
    ap.add_argument(
        "--compact_header_stub",
        action="store_true",
        help="For compact 5-field pair rows without explicit diagrams, build a "
             "single-vertex stub diagram from each process header (externals only). "
             "Prefer diagram sidecars or full 7-field exports when available.",
    )
    ap.add_argument("--mass_rewrite_mode", choices=["species", "leg_sets"], default="species",
                    help="How mass tokens in the target are canonicalized. "
                         "'species' (default) assigns slots per distinct species in X_id order, "
                         "producing compact tokens like M1, M2, Mi1. "
                         "'leg_sets' assigns slots keyed by the sorted set of external legs that share "
                         "the species, producing tokens like M_1_3 that directly bind to the encoder's "
                         "anchor indices. Pick 'leg_sets' when you suspect the decoder is paying a "
                         "sample-efficiency tax learning the implicit anchor-species mapping.")
    ap.add_argument("--mass_rewrite", action=argparse.BooleanOptionalAction, default=True,
                    help="Rewrite m_<species> tokens in target expressions to diagram-derived "
                         "slot names (M1, M2, ...). Enabled by default; disable with --no-mass_rewrite "
                         "to train on literal particle-name masses.")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accumulate_grad_batches", type=int, default=1,
                    help="Number of micro-batches per optimizer step (Lightning gradient "
                         "accumulation). Effective batch size is batch_size * this value. "
                         "Use e.g. --batch_size 2 --accumulate_grad_batches 16 to mimic batch 32 on a small GPU.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--scheduler", choices=["none","cosine","cosine_warmup","linear_warmup","step",
                    "reduce_on_plateau","onecycle","cosine_warm_restarts"], default="cosine_warmup")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--step_size", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--t_0", type=int, default=10,
                    help="Number of epochs for the first restart cycle (CosineAnnealingWarmRestarts T_0).")
    ap.add_argument("--t_mult", type=int, default=2,
                    help="Factor to increase cycle length after each restart (CosineAnnealingWarmRestarts T_mult).")
    ap.add_argument("--eta_min", type=float, default=0.0,
                    help="Minimum learning rate for cosine annealing schedulers.")
    ap.add_argument("--loss_mode", choices=["ce", "ctc"], default="ce",
        help=(
            "Training objective for the token decoder. "
            "'ctc' = deterministic full-sequence decoding via best-path CTC (collapse repeats, drop blanks/PAD). "
            "'ce'  = position-aligned cross-entropy."
        ),
    )
    ap.add_argument(
        "--output_mode",
        choices=["sequence", "term_slots"],
        default="term_slots",
        help=(
            "Output representation: 'term_slots' (default) predicts canonical "
            "structured algebraic slots with CE; 'sequence' predicts the flat "
            "canonical token sequence and uses --loss_mode."
        ),
    )
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--length_loss_weight", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_seq_len_cap", type=int, default=4096)
    ap.add_argument("--fail_log_dir", type=str, default=None,
                    help="Directory to append JSONL logs of seq-accuracy failures. Disabled if not set.")
    ap.add_argument("--fail_log_max", type=int, default=64,
                    help="Max failed examples to log per epoch per stage.")

    ap.add_argument("--enc_hid", type=int, default=128,
                    help="Graph encoder hidden size.")
    ap.add_argument("--enc_layers", type=int, default=None,
                    help="Graph encoder TransformerConv layers. If omitted, use model default.")
    ap.add_argument("--enc_heads", type=int, default=None,
                    help="Graph encoder attention heads. If omitted, use model default.")
    ap.add_argument("--enc_dropout", type=float, default=None,
                    help="Graph encoder dropout. If omitted, use model default.")
    ap.add_argument("--d_model", type=int, default=None,
                    help="Projection + token-decoder model dimension. If omitted, defaults to 256.")
    ap.add_argument("--dec_nhead", type=int, default=None,
                    help="Token decoder attention heads. If omitted, use model default.")
    ap.add_argument("--dec_layers", type=int, default=None,
                    help="Token decoder transformer layers. If omitted, use model default.")
    ap.add_argument("--dec_dropout", type=float, default=None,
                    help="Token decoder dropout. If omitted, use model default.")
    ap.add_argument("--dec_max_len", type=int, default=None,
                    help="Token decoder max positional length. If omitted, use model default.")
    ap.add_argument("--dec_use_len_mask", action="store_true",
        help=(
            "If set, length-mask the decoder's non-causal self-attention so padded/unneeded "
            "positions cannot influence valid positions. Training still uses gold lengths from "
            "y_in; val/test accuracy uses the length head and generate() (predicted length). "
            "If unset, the model self-terminates via [EOS]."
        ),
    )
    ap.add_argument("--slot_max_exponent", type=int, default=8)
    ap.add_argument("--slot_num_slots", type=int, default=64)
    ap.add_argument("--slot_num_factors", type=int, default=3)
    ap.add_argument("--slot_den_slots", type=int, default=4)
    ap.add_argument("--slot_den_terms", type=int, default=5)
    ap.add_argument("--slot_den_term_factors", type=int, default=3)
    ap.add_argument(
        "--slot_degree_loss_weight",
        type=float,
        default=0.1,
        help="Weight of topology-derived expected-degree consistency for term slots.",
    )
    ap.add_argument(
        "--slot_no_object_weight",
        type=float,
        default=0.1,
        help="CE weight for inactive canonical term/factor slots.",
    )
    ap.add_argument("--phys_decode", choices=["none", "masses"], default="none",
                    help="Physics-informed decoding: mask logits for mass tokens not allowed "
                         "by the diagram slot_map. Applied consistently in train/val/test.")
    ap.add_argument(
        "--simplify_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SymPy-simplify mass-rewritten targets before canonicalization "
             "(train/val/test all use the same simplified targets).",
    )
    ap.add_argument("--subsample", type=int, default=None,
                    help="Randomly keep only N dataset lines (seeded by --seed) before the "
                         "train/val/test split. Useful for quick experiments, e.g. testing "
                         "--simplify_targets on a subset without hours of SymPy prep.")
    ap.add_argument("--canonicalize_targets", action=argparse.BooleanOptionalAction, default=True,
                    help="Canonicalize commutative mass/Mandelstam factor order in dataset targets.")
    ap.add_argument("--canonicalize_eval", action=argparse.BooleanOptionalAction, default=True,
                    help="Compare val/test predictions using commutative canonical form "
                         "(MDE^2 MAC^2 == MAC^2 MDE^2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Independent seed for the train/validation/test row split.",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.90,
        help="Fraction assigned to training within each model/order/rank stratum.",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction assigned to validation; the remainder is assigned to test.",
    )
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    ap.add_argument("--log_every_n_steps", type=int, default=50)
    ap.add_argument("--print_every_n_epochs", type=int, default=0,
                    help="If >0, print one compact rank-zero metric line every N validation epochs.")
    ap.add_argument("--enable_progress_bar", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--enable_model_summary", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--enable_logger", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=True,
                    help="If set (default), look for last.ckpt/latest *.ckpt in --out_dir "
                         "and resume from it when --resume_from is not given. "
                         "Refuses to resume if the stored run_state fingerprint disagrees "
                         "with the current data/model shape. Disable with --no-auto_resume.")
    ap.add_argument("--ckpt_every_n_steps", type=int, default=0,
                    help="If >0, additionally save a checkpoint every N training steps. "
                         "Useful for long-running jobs (e.g. EW 2_to_3) where a full epoch "
                         "may not fit inside the job window.")
    ap.add_argument("--test_only", action="store_true",
                    help="Skip training and only run test on the specified checkpoint (best or path).")
    ap.add_argument("--max_perms", type=int, default=12,
                help="Fixed number of diagram permutations per sample (preallocated).")
    ap.add_argument("--compile", action="store_true",
                help="Apply torch.compile to the model for faster training (slow first step).")
    args = ap.parse_args()

    if args.output_mode == "term_slots":
        if args.loss_mode != "ce":
            ap.error("--output_mode term_slots requires --loss_mode ce")
        if not args.simplify_targets:
            ap.error("--output_mode term_slots requires --simplify_targets")
        if args.dec_use_len_mask:
            ap.error("--dec_use_len_mask applies only to sequence output")
        if args.phys_decode != "none":
            ap.error("--phys_decode token masks do not apply to structured slots")
        if args.slot_degree_loss_weight < 0:
            ap.error("--slot_degree_loss_weight must be non-negative")
        if args.slot_no_object_weight <= 0:
            ap.error("--slot_no_object_weight must be positive")

    if args.output_mode == "sequence" and args.dec_max_len is None and args.max_seq_len_cap is not None:
        args.dec_max_len = int(args.max_seq_len_cap)
    elif (
        args.output_mode == "sequence"
        and
        args.dec_max_len is not None
        and args.max_seq_len_cap is not None
        and int(args.dec_max_len) < int(args.max_seq_len_cap)
    ):
        print(
            f"[WARN] --dec_max_len ({args.dec_max_len}) < --max_seq_len_cap ({args.max_seq_len_cap}); "
            "setting dec_max_len=max_seq_len_cap to avoid truncation."
        )
        args.dec_max_len = int(args.max_seq_len_cap)
    elif args.output_mode == "sequence" and args.dec_max_len is None and args.max_seq_len_cap is None:
        print(
            "[WARN] --max_seq_len_cap is None and --dec_max_len is None; decoder max_len defaults to 4096. "
            "If your dataset contains longer targets, training will truncate/break."
        )

    pl.seed_everything(args.seed)

    orders = None if args.orders == "all" else [s.strip() for s in args.orders.split(",") if s.strip()]
    ranks = None if not args.ranks else [s.strip() for s in args.ranks.split(",") if s.strip()]
    model_list = _parse_model_list(args.model)
    print(f"[data] training on physics models: {model_list}")

    train_ds, val_ds, test_ds, vocab, rank_anchor_dims = build_datasets(
        model_list, args.data_dir,
        n_initial=args.n_initial,
        max_seq_len_cap=args.max_seq_len_cap,
        max_perms=args.max_perms,
        orders=orders, ranks=ranks,
        mass_rewrite=args.mass_rewrite,
        mass_rewrite_mode=args.mass_rewrite_mode,
        canonicalize_targets=args.canonicalize_targets,
        compact_header_stub=args.compact_header_stub,
        simplify_targets=args.simplify_targets,
        subsample=args.subsample,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        output_mode=args.output_mode,
        slot_max_exponent=args.slot_max_exponent,
        slot_num_slots=args.slot_num_slots,
        slot_num_factors=args.slot_num_factors,
        slot_den_slots=args.slot_den_slots,
        slot_den_terms=args.slot_den_terms,
        slot_den_term_factors=args.slot_den_term_factors,
    )

    term_slot_layout = None
    if args.output_mode == "term_slots":
        term_slot_layout = {
            "num_slots": args.slot_num_slots,
            "num_factors": args.slot_num_factors,
            "den_slots": args.slot_den_slots,
            "den_terms": args.slot_den_terms,
            "den_term_factors": args.slot_den_term_factors,
        }
    dm = GraphExprDataModule(
        (train_ds, val_ds, test_ds),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_perms=args.max_perms,
        pad_id=vocab.pad,
        term_slot_layout=term_slot_layout,
    )
    if args.phys_decode != "none":
        print(f"[train] physics-informed decode mask: {args.phys_decode}")
    if args.canonicalize_targets:
        print("[train] commutative target canonicalization: enabled")
    if args.simplify_targets:
        print("[train] sympy target simplification: enabled (dataset prep is CPU-heavy)")
    if args.output_mode == "term_slots":
        print(
            "[train] structured term-slot CE: direct canonical-slot supervision + "
            f"topology degree weight {args.slot_degree_loss_weight:g}"
        )
    if args.output_mode == "sequence" and args.canonicalize_eval:
        print("[train] commutative eval matching: enabled")
    if args.compact_header_stub:
        print("[train] compact 5-field rows: using process-header stub diagrams")
    if args.accumulate_grad_batches > 1:
        eff = args.batch_size * args.accumulate_grad_batches
        print(f"[train] gradient accumulation: {args.accumulate_grad_batches} steps "
              f"(effective batch size {eff} = {args.batch_size} x {args.accumulate_grad_batches})")

    graphs0, *_rest = train_ds[0]
    sample_graph = next(g for g in graphs0 if g is not None)
    in_dim = sample_graph.x.size(1)
    edge_dim = (
        sample_graph.edge_attr.size(1)
        if getattr(sample_graph, "edge_attr", None) is not None and sample_graph.edge_attr.numel() > 0
        else 1
    )
    graph_features = getattr(sample_graph, "graph_features", None)
    graph_feat_dim = int(graph_features.size(-1)) if graph_features is not None else 0
    if graph_feat_dim != GRAPH_FEAT_DIM:
        raise ValueError(
            f"expected {GRAPH_FEAT_DIM} graph-level physics features, got {graph_feat_dim}"
        )

    x0 = sample_graph.x
    is_external = x0[:, 0] > 0.5
    is_amp_side = x0[:, -1] < 0.5
    num_anchors = int((is_external & is_amp_side).sum().item())
    slot_vocab = getattr(vocab, "term_slot_vocabulary", None)

    lit = LitAmpGNN(
        in_dim=in_dim, edge_dim=edge_dim, graph_feat_dim=graph_feat_dim,
        vocab_size=len(vocab.stoi), pad_id=vocab.pad, bos_id=vocab.bos, eos_id=vocab.eos,
        lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, warmup_steps=args.warmup_steps,
        t_0=args.t_0, t_mult=args.t_mult, eta_min=args.eta_min,
        loss_mode=args.loss_mode,
        label_smoothing=args.label_smoothing, length_loss_weight=args.length_loss_weight, max_steps=args.max_steps,

        enc_hid=args.enc_hid, enc_layers=args.enc_layers, enc_heads=args.enc_heads, enc_dropout=args.enc_dropout,
        d_model=args.d_model, dec_nhead=args.dec_nhead, dec_layers=args.dec_layers,
        dec_dropout=args.dec_dropout, dec_max_len=args.dec_max_len, dec_use_len_mask=args.dec_use_len_mask,
        output_mode=args.output_mode,
        slot_coefficient_size=slot_vocab.coefficient_size if slot_vocab is not None else 0,
        slot_symbol_size=slot_vocab.symbol_size if slot_vocab is not None else 0,
        slot_coupling_size=slot_vocab.coupling_size if slot_vocab is not None else 0,
        slot_max_exponent=args.slot_max_exponent,
        slot_num_slots=args.slot_num_slots,
        slot_num_factors=args.slot_num_factors,
        slot_den_slots=args.slot_den_slots,
        slot_den_terms=args.slot_den_terms,
        slot_den_term_factors=args.slot_den_term_factors,
        slot_degree_loss_weight=args.slot_degree_loss_weight,
        slot_no_object_weight=args.slot_no_object_weight,
        slot_symbol_degree2=slot_vocab.symbol_degree2() if slot_vocab is not None else None,
        slot_zero_coefficient_id=(
            slot_vocab.coefficient_to_id.get("0", -1) if slot_vocab is not None else -1
        ),
        num_anchors=num_anchors,
        rank_anchor_dims=rank_anchor_dims if len(rank_anchor_dims) > 1 else None,
        failure_log_dir=args.fail_log_dir, failure_log_max=args.fail_log_max,
    )

    lit.attach_vocab(vocab)
    lit.phys_decode_mode = str(args.phys_decode)
    lit.canonicalize_commutative = bool(
        args.output_mode == "sequence" and args.canonicalize_eval
    )

    if args.compile:
        lit.model = torch.compile(lit.model, dynamic=True)


    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = "_".join(model_list) if len(model_list) > 1 else model_list[0]
    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir,
        filename=f"{model_tag}" + "-{epoch:02d}-{val_seq_acc:.4f}",
        save_top_k=1,
        monitor="val/seq_acc",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    cbs = [ckpt_cb]
    if args.print_every_n_epochs and args.print_every_n_epochs > 0:
        cbs.append(CompactEpochPrinter(every_n_epochs=args.print_every_n_epochs))
    if args.enable_logger:
        cbs.append(LearningRateMonitor(logging_interval="step"))
    if args.ckpt_every_n_steps and args.ckpt_every_n_steps > 0:
        step_ckpt_cb = ModelCheckpoint(
            dirpath=args.out_dir,
            filename=f"{model_tag}-step" + "-{step}",
            every_n_train_steps=int(args.ckpt_every_n_steps),
            save_top_k=1,
            save_last=True,
            monitor=None,
            auto_insert_metric_name=False,
        )
        cbs.append(step_ckpt_cb)
    if args.early_stop:
        cbs.append(EarlyStopping(monitor="val/seq_acc", patience=args.patience, mode="max"))


    run_state = build_run_state(args, vocab, in_dim=in_dim, edge_dim=edge_dim,
                                graph_feat_dim=graph_feat_dim,
                                num_anchors=num_anchors,
                                rank_anchor_dims=rank_anchor_dims)
    resume_ckpt: Optional[str] = args.resume_from
    if resume_ckpt is None and args.auto_resume and not args.test_only:
        discovered = discover_resume_ckpt(args.out_dir)
        if discovered is not None:
            ok, why = check_run_state_compatible(args.out_dir, run_state)
            if ok:
                print(f"[AUTO-RESUME] resuming from {discovered} ({why})")
                resume_ckpt = discovered
            else:
                print(
                    f"[AUTO-RESUME] refused to resume from {discovered}:\n{why}\n"
                    "Pass --no-auto_resume to force a fresh start, or --resume_from PATH to override.",
                    file=sys.stderr,
                )
                sys.exit(3)
        else:
            print(f"[AUTO-RESUME] no checkpoint found in {args.out_dir}, starting fresh.")

    write_run_state(args.out_dir, run_state)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.out_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=cbs,
        enable_checkpointing=True,
        enable_progress_bar=args.enable_progress_bar,
        enable_model_summary=args.enable_model_summary,
        logger=args.enable_logger,
        deterministic=False,
    )

    if args.test_only:

        ckpt_path = args.resume_from if args.resume_from is not None else "best"
        test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt_path)
        if trainer.is_global_zero:
            write_result_json(
                args.out_dir,
                "test_metrics.json",
                {"checkpoint": ckpt_path, "results": test_results},
            )
        return


    trainer.fit(lit, datamodule=dm, ckpt_path=resume_ckpt)

    if trainer.is_global_zero:
        best_score = ckpt_cb.best_model_score
        write_result_json(
            args.out_dir,
            "fit_result.json",
            {
                "best_model_path": ckpt_cb.best_model_path,
                "best_model_score": (
                    float(best_score.detach().cpu().item())
                    if best_score is not None
                    else None
                ),
                "epochs_completed": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "parameter_count": int(sum(p.numel() for p in lit.parameters())),
            },
        )

    test_results = trainer.test(lit, datamodule=dm, ckpt_path="best")
    if trainer.is_global_zero:
        write_result_json(
            args.out_dir,
            "test_metrics.json",
            {"checkpoint": ckpt_cb.best_model_path, "results": test_results},
        )

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
