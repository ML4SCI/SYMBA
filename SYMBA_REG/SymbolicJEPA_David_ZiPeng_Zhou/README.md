# JEPA for Symbolic Regression

This project explores Joint Embedding Predictive Architecture (JEPA) pretraining for SymbolicGPT-style symbolic regression. The model receives a numerical point cloud sampled from an unknown function and autoregressively generates a symbolic expression in prefix notation.

Two JEPA pretraining methods are included.

**Numeric-Symbolic JEPA** aligns the numerical representation produced by the T-Net encoder with a representation of the target symbolic expression.

**Subsample JEPA** trains the numerical encoder to produce consistent representations for independently sampled point clouds from the same underlying function.

After JEPA pretraining, both models are trained using the same standard SymbolicGPT cross-entropy objective. This keeps downstream training fixed and isolates the effect of representation pretraining.

## Repository structure

```text
JEPA/
├── README.md
├── requirements.txt
├── numeric_symbolic_jepa.ipynb
├── subsample_jepa.ipynb
│
├── data/
│   ├── README.md
│   ├── 01_generate_synthetic_data.ipynb
│   ├── 02_build_template_dataset.ipynb
│   └── synthetic_templates.pkl
│
└── symbolic_jepa/
    └── ...
```

The `symbolic_jepa` package contains the shared model, tokenizer, dataset, evaluation, checkpointing, and JEPA infrastructure used by the experiment notebooks.

## Running in Google Colab

Google Colab with a GPU runtime is the recommended environment.

To run an experiment, download the desired notebook from this repository and upload it directly to Google Colab.

```text
numeric_symbolic_jepa.ipynb
```

or

```text
subsample_jepa.ipynb
```

Select a GPU under

```text
Runtime > Change runtime type > GPU
```

The notebook handles the remaining environment setup, including mounting Google Drive, cloning or updating the repository, and loading the required project files.

The required Python packages are listed in `requirements.txt`. The notebook can install them after the repository has been cloned.

## Google Drive

Google Colab runtimes are temporary, so files stored only in the runtime are lost when the session is reset. The notebooks therefore use Google Drive as persistent storage.

The first time an experiment is run, the notebook mounts Google Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

The repository is then cloned into a directory on Google Drive. Subsequent sessions can reuse the same checkout instead of cloning the project again.

A typical layout is

```text
MyDrive/
└── Symba/
    └── symbolic-jepa/
```

Keeping the repository on Drive also makes generated data and experiment configuration available across Colab sessions.

Google Drive is also used to preserve training checkpoints. This is particularly important for longer experiments because Colab runtimes may disconnect before training is complete. Saved checkpoints allow training to resume in a later session rather than restarting from the beginning.

Depending on the notebook configuration, the Drive directory may contain

```text
repository
datasets
checkpoints
TensorBoard logs
evaluation results
```

The notebook setup cells define the relevant Drive and repository paths. Users can change these paths if they prefer a different location in their own Google Drive.

## Dataset

The experiments use synthetically generated single-variable mathematical expressions.

A prepared canonical template dataset is included as

```text
data/synthetic_templates.pkl
```

This file is sufficient to run the two final experiment notebooks.

The complete dataset-generation process can also be reproduced from scratch using the two notebooks in the `data` directory.

### 1. Generate the raw synthetic dataset

Run

```text
data/01_generate_synthetic_data.ipynb
```

This generates

```text
data/synthetic.pkl
```

containing instantiated symbolic expressions with sampled numerical constants.

### 2. Build the canonical template dataset

Next, run

```text
data/02_build_template_dataset.ipynb
```

This converts the raw expressions into canonical symbolic templates and constructs the coefficient pools used for dynamic coefficient augmentation.

It produces

```text
data/synthetic_templates.pkl
```

Users who only want to run the reported experiments can use the included template dataset and skip these two steps.

## Numeric-Symbolic JEPA

The Numeric-Symbolic experiment is contained in

```text
numeric_symbolic_jepa.ipynb
```

The numerical encoder is pretrained by aligning its point-cloud representation with a stop-gradient representation of the corresponding symbolic expression.

After pretraining, the JEPA objective is removed and the complete model is trained using the standard autoregressive symbolic regression objective.

The reported experiments compare 0, 10, and 20 epochs of JEPA pretraining.

## Subsample JEPA

The Subsample experiment is contained in

```text
subsample_jepa.ipynb
```

Two independently sampled point clouds from the same underlying function are encoded by the numerical encoder. JEPA pretraining encourages their centered representations to agree.

The downstream model is then trained using the same SymbolicGPT objective as the Numeric-Symbolic experiment.

The reported experiments compare 0 and 10 epochs of Subsample JEPA pretraining.

## Evaluation

The experiments use several complementary metrics.

**Validation token accuracy** measures teacher-forced next-token prediction accuracy.

**Branch accuracy** measures accuracy at positions in the prefix tree where more than one continuation is possible.

**Exact match** requires the complete generated prefix expression to exactly match the target.

**Functional equivalence** checks whether the generated expression represents the same function using symbolic comparison followed by numerical verification when necessary.

**Numerical accuracy** fits the constants of the predicted expression and evaluates it on a separate point cloud. Results report the fraction of predictions with \(R^2 > 0.9\).

The first two metrics are teacher-forced diagnostics. Exact match, functional equivalence, and numerical accuracy evaluate complete autoregressively generated expressions.

## Checkpoints and TensorBoard

The experiment notebooks periodically save checkpoints during training. Checkpoints stored on Google Drive persist after the Colab runtime disconnects and can be used to resume an incomplete experiment.

TensorBoard logs are also saved during training and can be viewed in Colab with

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
