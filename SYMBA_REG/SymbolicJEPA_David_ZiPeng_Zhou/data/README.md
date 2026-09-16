# Dataset Generation

The experiments use `synthetic_templates.pkl`, which contains the canonical symbolic templates and coefficient pools used during training.

The included template dataset can be used directly. To reproduce it from scratch, run the following two steps in order.

## 1. Generate the raw expressions

Run

```text
01_SYMBA_Reg_Data_Gen.ipynb
```

This produces

```text
synthetic.pkl
```

containing the generated symbolic expressions.

## 2. Build the canonical templates

From the project root, run

```bash
python data/02_build_template_dataset.py \
    --input data/synthetic.pkl \
    --output data/synthetic_templates.pkl \
    --max-expressions 200000 \
    --max-vars 1
```

This produces

```text
synthetic_templates.pkl
```

which is the dataset used by `numeric_symbolic.ipynb` and `subsample.ipynb`.

The raw `synthetic.pkl` file is not required if the included `synthetic_templates.pkl` is used directly.
