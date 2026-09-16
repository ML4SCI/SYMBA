# JEPA for Symbolic Regression

This project explores Joint Embedding Predictive Architecture (JEPA) pretraining for SymbolicGPT-style symbolic regression. The model receives a numerical point cloud sampled from an unknown function and autoregressively generates a symbolic expression in prefix notation.

Two JEPA pretraining methods are included.

**Numeric-Symbolic JEPA** aligns the numerical representation produced by the T-Net encoder with a representation of the target symbolic expression.

**Subsample JEPA** trains the numerical encoder to produce consistent representations for independently sampled point clouds from the same underlying function.

After JEPA pretraining, both models are trained using the standard SymbolicGPT cross-entropy objective.

## Project Writeup

A detailed overview of the motivation, methods, experiments, and results is available in the accompanying Medium article:

[JEPA for Symbolic Regression](https://medium.com/@zzpdavid2l/jepa-for-symbolic-regression-791843364837?sharedUserId=zzpdavid2l)

## Repository structure

```text
SymbolicJEPA_David_ZiPeng_Zhou/
├── README.md
├── requirements.txt
├── numeric_symbolic.ipynb
├── subsample.ipynb
│
├── data/
│   ├── README.md
│   ├── 01_SYMBA_Reg_Data_Gen.ipynb
│   ├── 02_build_template_dataset.py
│   └── synthetic_templates.pkl
│
└── symbolic_jepa/
    └── ...
```

The `symbolic_jepa` package contains the shared model, tokenizer, dataset, evaluation, checkpointing, and JEPA infrastructure used by the experiment notebooks.

## Running in Google Colab

Google Colab with a GPU runtime is the recommended environment.

To run an experiment, download the desired notebook and upload it directly to Google Colab.

```text
numeric_symbolic.ipynb
```

or

```text
subsample.ipynb
```

Select a GPU under

```text
Runtime > Change runtime type > GPU
```

The notebook handles the remaining environment setup, including mounting Google Drive, cloning or updating the repository, and loading the project files.

The required Python packages are listed in `requirements.txt`.

## Google Drive

Google Colab runtimes are temporary, so files stored only in the runtime are lost when the session ends. The experiment notebooks use Google Drive to preserve the repository and experiment state across Colab sessions.

The notebook first mounts Google Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

The repository checkout is stored under Google Drive so it can be reused in later sessions. The default setup uses

```text
/content/drive/MyDrive/Symba/
```

The notebooks also use Google Drive for persistent checkpoint backups. Training itself may use local Colab storage for faster file access, while checkpoints are periodically copied to Drive so an interrupted experiment can be resumed later.

The Drive directory may therefore contain the repository, generated data, checkpoint backups, TensorBoard logs, and evaluation results.

The relevant paths are defined near the beginning of each notebook and can be changed if a different Drive layout is preferred.

## Dataset

The experiments use synthetically generated single-variable mathematical expressions.

A prepared canonical template dataset is included as

```text
data/synthetic_templates.pkl
```

This file is sufficient to run the final experiment notebooks.

The complete data-generation pipeline can also be reproduced from scratch.

### 1. Generate the raw synthetic dataset

Run

```text
data/01_SYMBA_Reg_Data_Gen.ipynb
```

This generates

```text
data/synthetic.pkl
```

containing instantiated symbolic expressions with sampled numerical constants.

### 2. Build the canonical template dataset

From the project root, run

```bash
python data/02_build_template_dataset.py \
    --input data/synthetic.pkl \
    --output data/synthetic_templates.pkl \
    --max-expressions 200000 \
    --max-vars 1
```

This converts the raw expressions into canonical symbolic templates and constructs the coefficient pools used for dynamic coefficient augmentation.

Users who only want to run the reported experiments can use the included `synthetic_templates.pkl` and skip dataset generation.

## Numeric-Symbolic JEPA

The Numeric-Symbolic experiment is contained in

```text
numeric_symbolic.ipynb
```

The numerical encoder is pretrained by aligning its point-cloud representation with a stop-gradient representation of the corresponding symbolic expression.

After pretraining, the JEPA objective is removed and the complete model is trained using the standard autoregressive symbolic regression objective.

The reported experiments compare 0, 10, and 20 epochs of JEPA pretraining.

## Subsample JEPA

The Subsample experiment is contained in

```text
subsample.ipynb
```

Two independently sampled point clouds from the same underlying function are encoded by the numerical encoder. JEPA pretraining encourages their centered representations to agree.

The downstream model is then trained using the same SymbolicGPT objective as the Numeric-Symbolic experiment.

The reported Subsample experiment compares 0 and 20 epochs of Subsample JEPA pretraining.

## Evaluation

The experiments use several complementary metrics.

**Validation token accuracy** measures teacher-forced next-token prediction accuracy.

**Branch accuracy** measures accuracy at positions in the prefix tree where more than one continuation is possible.

**Exact match** requires the complete generated prefix expression to exactly match the target.

**Functional equivalence** checks whether the generated expression represents the same function using symbolic comparison followed by numerical verification when necessary.

**Numerical accuracy** fits the constants of the predicted expression and evaluates it on a separate point cloud. Results report the fraction of predictions with \(R^2 > 0.9\).

The first two metrics are teacher-forced diagnostics. Exact match, functional equivalence, and numerical accuracy evaluate complete autoregressively generated expressions.

## Checkpoints and TensorBoard

The experiment notebooks periodically save checkpoints during training.

In Colab, active checkpoints can be stored on the local runtime for faster access while persistent backup copies are stored on Google Drive. If the runtime disconnects, the saved Drive checkpoint can be restored when the experiment is resumed.

TensorBoard logs are also produced during training and can be viewed in Colab with

```python
%load_ext tensorboard
%tensorboard --logdir PATH_TO_LOGS
```

## Reproducing the reported experiments

For exact reproduction, keep the model seed, data seed, dataset split, pretraining duration, and downstream training configuration unchanged.

The Numeric-Symbolic experiments separate model initialization randomness from data-generation randomness. This allows matched comparisons in which different pretraining conditions receive the same downstream data trajectory.

Validation and test sets are deterministic. Canonical symbolic structures are split before numerical coefficients are sampled, preventing different numerical realizations of the same symbolic structure from appearing across the training and evaluation splits.

## Acknowledgements

This work was developed as part of ML4SCI and builds on SymbolicGPT-style symbolic regression and the Joint Embedding Predictive Architecture framework.
