# Physics-blind baseline

A comparison baseline for the physics-informed AmpGNN. It reads the same Feynman
diagram graphs and predicts the same squared-amplitude expressions, but every
physics prior has been removed from the input, so the accuracy gap between the
two measures what the physics knowledge is actually worth.

Nothing in the parent package is modified. Everything that should be held
constant is imported from it, so the two models are compared on exactly the same
footing:

| Shared with the main model | Source |
| --- | --- |
| Data loading and row parsing | `ampgnn.data_loading` |
| Seeded stratified train/val/test split | `ampgnn.train._stratified_split` |
| Target tokenisation, SymPy simplification, canonicalisation | `ampgnn.expr_simplify`, `ampgnn.expr_canonical` |
| Vocabulary | `ampgnn.tokenizer.Vocab` |
| Non-autoregressive decoder, length head | `ampgnn.model.TokenDecoder`, `AmpGNN` |
| CE/CTC loss, accuracy metrics, failure logging, optimiser and scheduler | `ampgnn.model.LitAmpGNN` |
| Batching, rank bucketing, epoch printing | `ampgnn.train` |

## What is removed

| Physics prior in the main model | Baseline instead gets |
| --- | --- |
| Hand-coded particle vectors (spin, charge, T3, hypercharge, colour, generation, fermion number, self-conjugacy, chirality) | One-hot over the raw particle symbol, with separate slots for particle vs. antiparticle |
| Amplitude↔conjugate cross edges encoding the interference leg pairing | Both diagrams in one graph, distinguished only by a side flag |
| Anchor-distance features (hop distances to amplitude-side external legs) | Nothing — plain topology and the degree encoding |
| Leg-permutation augmentation over identical outgoing particles | Identity pairing only |
| `--phys_decode masses` allowlist at decode time | No decode constraint |
| Mass rewriting to diagram-derived leg slots (`MAB`, `MABDE`, ...) | Raw species (`m_u`, `m_d`, ...), controlled by `--mass_rewrite` |

The graph topology itself is kept: vertices, propagators, external attachments
and node degrees are all still visible. The baseline can read the structure of
the diagram, it just has no idea what the particles physically are.

Dropping the mass rewrite is a 1:1 token substitution, so target lengths are
unchanged (mean 426.8 tokens on QED 2→3 either way) and the vocabulary grows by
one entry. Sequence accuracy therefore stays broadly comparable; the model
simply has to infer the species instead of being handed the leg alignment.

## Files

| File | Contents |
| --- | --- |
| `graph_blind.py` | `BlindFeaturizer`, `build_blind_base_graph`, `build_blind_pair_graphs` |
| `data.py` | `build_blind_datasets` — same split and targets, blind graphs |
| `model.py` | `LitPhysicsBlind` — `LitAmpGNN` with the anchor-free encoder |
| `train.py` | Training CLI |
| `seed_compare.py` | Runs both models across seeds and reports mean ± stdev |

## Usage

Train the baseline:

```
python -m ampgnn.physics_blind_model.train \
  --model QED --data_dir ampgnn/data_july2026 \
  --out_dir ampgnn/runs/qed_2to2_blind \
  --orders tree --ranks 2_to_2 \
  --epochs 100 --batch_size 16 --lr 5e-4 \
  --label_smoothing 0.05 --gradient_clip_val 0.5 --weight_decay 0.01 \
  --accelerator gpu --devices 1 --num_workers 4
```

Run the head-to-head comparison across seeds:

```
python -m ampgnn.physics_blind_model.seed_compare \
  --model QED --ranks 2_to_2 --seeds 42 43 44 --epochs 100 \
  --results_dir ampgnn/runs/baseline_seed_compare_qed_2to2
```

The seed controls the split as well as initialisation, so re-running with more
seeds widens the coverage; completed runs are skipped on re-invocation.

## Results so far

QED 2→2, tree level, 100 epochs, 3 seeds (42/43/44), identical hyperparameters
(`label_smoothing=0.05`, `gradient_clip_val=0.5`, `weight_decay=0.01`):

| | test seq acc | test token acc |
| --- | --- | --- |
| Physics-informed | 0.829 ± 0.039 | 0.846 ± 0.023 |
| Physics-blind | 0.350 ± 0.030 | 0.739 ± 0.014 |

Token accuracy falls by only 11 points while sequence accuracy collapses by 48:
the baseline learns the general form of the expressions but rarely produces one
that is exactly right.

> The test split is ~39 examples for QED 2→2, so each example is worth 2.6
> points and single runs swing by several points (PyG scatter ops are
> non-deterministic on CUDA and the trainer runs with `deterministic=False`).
> Report the mean over seeds rather than a single run.

## Caveats

Note that `--no-mass_rewrite` changes the expression strings, so a
`AMPGNN_SIMPLIFY_CACHE` built from mass-rewritten targets misses on every
lookup. Point the baseline at its own cache file and expect a slow first SymPy
pass.

`--dec_use_len_mask` (on by default here, matching the main model) leaks the
gold sequence length at evaluation time. That applies equally to both arms, so
the comparison is fair, but the absolute numbers are optimistic.
