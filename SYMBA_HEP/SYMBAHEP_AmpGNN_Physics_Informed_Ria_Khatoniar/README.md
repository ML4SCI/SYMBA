# AmpGNN: Physics-Informed Graph Networks for Squared-Amplitude Terms

Contributor: **Ria Khatoniar**

This folder contains **AmpGNN**, a physics-informed graph neural network for predicting
symbolic squared-amplitude *terms* from Feynman diagram pairs (QED/QCD, tree-level
\(2\to2\) and \(2\to3\)).

Unlike sequence-only SYMBA-style amplitude \(\to\) squared-amplitude models, AmpGNN:

- represents each amplitude–conjugate **diagram pair** as a graph;
- injects Standard Model particle properties, interference cross edges, anchor
  distances, and related structural priors;
- supports a parallel **sequence** decoder and a **structured term-slot** decoder;
- includes a **physics-blind** ablation baseline that keeps topology but removes
  physics-specific features.

Upstream development lives in the AmpGNN repository; this snapshot is contributed to
[SYMBA-HEP](https://github.com/ML4SCI/SYMBA/tree/main/SYMBA_HEP) for the ML4SCI /
SYMBA project collection.

## Layout

```
ampgnn/
  train.py                  # physics-informed training entry
  model.py                  # GNN encoder + NAR / term-slot decoders
  graph_builder.py          # pair graphs + physics features
  physics_blind_model/      # ablation baseline
  term_slots.py             # structured target decomposition
  scripts/                  # simplify / precompute helpers
  tests/
```

## Quick start

```bash
# from this folder
pip install -r ampgnn/requirements.txt
# plus torch / pytorch-lightning / torch-geometric matching your CUDA setup

# run as a package (add parent of `ampgnn/` to PYTHONPATH)
cd ..
PYTHONPATH=. python -m ampgnn.train --help
PYTHONPATH=. python -m ampgnn.physics_blind_model.train --help
```

Point `--data_dir` at MARTY diagram-pair exports (see `ampgnn/README.md` for the
`pair i-j` row format). Large datasets, run directories, and SQLite caches are not
bundled here.

## Citation / related SYMBA work

See the main [SYMBA-HEP README](../README.md) and prior Transformer / SSM contributions
in sibling folders under `SYMBA_HEP/`.
