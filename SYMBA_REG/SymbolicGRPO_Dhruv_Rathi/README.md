# SymbolicGRPO

GSoC contribution to ML4SCI's [SYMBA](https://ml4sci.org/gsoc/projects/2026/project_SYMBA.html) project.

A pretrained Transformer proposes candidate symbolic regression expressions that seed a genetic programming (GP) population, and GP search refines them from there. SymbolicGRPO swaps out the published preference pair refinement step (SymbolicDPO, which needs a frozen reference copy of the model) for a group relative update instead: sample 16 expressions per step, score each one with R-squared, and nudge the policy toward whichever samples beat their own group's mean. No reference copy needed.

## Results

Clean data, 3 seeds, 30 cells per method. Functional means R-squared >= 0.99.

| Method | Exact | Functional |
| --- | --- | --- |
| PIGP + SymbolicDPO | 16/30 | 21/30 |
| PIGP + SymbolicGRPO | 20/30 | 24/30 |

Gaussian noise added to training targets, 6 seeds, 60 cells per noise level:

| Noise sigma | DPO | GRPO |
| --- | --- | --- |
| 0.01 | 17/60 | 21/60 |
| 0.05 | 14/60 | 18/60 |
| 0.10 | 18/60 | 19/60 |

GRPO comes out ahead across the board, clean or noisy.

## Install

```bash
pip install -e .
```

## Run

```bash
python -m symba_grpo.runner --seed 42 --data-dir <path> --output-dir <path> --checkpoint <path>
```

The runner checks source and asset integrity against manifests in `sha256sum` format (`--source-manifest`, `--assets-manifest`, defaulting to `manifests/source.sha256` and `manifests/assets.sha256`). Generate your own with `sha256sum`.

## Built on

The [ML4SCI SYMBA project](https://ml4sci.org/gsoc/projects/2026/project_SYMBA.html), earlier SYMBA work by Aryamaan Thakur and Samyak Jha, and the AI-Feynman benchmark.
