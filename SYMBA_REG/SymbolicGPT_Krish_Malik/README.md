# SymbolicGPT: Symbolic Regression via T-Net Embeddings and GPT Decoder

This project implements symbolic regression using **T-Net embeddings**, a **GPT-style decoder**, and a **concept library** to model and rediscover scientific equations from the **Feynman dataset**.
It is part of the ML4SCI / Symba Lab work during **Google Summer of Code (GSoC)**.

---

## Project Structure


```
├── data/
│   ├── Feynman_csv_edit.csv          # Ground truth equations dataset
│   ├── data_cloud.py                 # Data cloud generation utilities
│   ├── data_clouds.json              # Pre-generated example data clouds
│   ├── feynman_parse_trees.json      # Parsed Feynman diagram trees
│   └── predictions.json              # Model predictions output (new)
│
├── src/
│   ├── parser/
│   │   └── symbolic_parser.py        # Core equation → parse tree converter
│   │
│   ├── embeddings/
│   │   ├── t_net_embeddings.py       # T-Net embedding implementation
│   │   ├── tnet_embeddings.json      # Pre-computed embeddings (legacy)
│   │   └── tnet_embeddings_new.json  # Improved embeddings (local focus)
│   │
│   ├── decoder/
│   │   ├── decoder.py                # GPT-style transformer decoder
│   │   ├── masking_decoder_setup.py  # Masked language modeling utilities
│   │   └── sliding_window.py         # Sliding-window sparse decoder (new)
│   │
│   ├── library/
│   │   └── learned_library.py        # Concept subtree library management
│   │
│   └── labels/
│       ├── tokenized_gpt_labels.json           # Baseline tokenized labels
│       └── tokenized_gpt_labels_with_full_funcs.json  # Enhanced labels with concept library
            # Medium blog post documentation
```

## Directory Overview  

#### `/data/`  
Contains all datasets, preprocessed artifacts, and model outputs. These form the foundation for training, evaluation, and analysis.  

- **Feynman_csv_edit.csv** → Core dataset of physics equations (Feynman equations). Serves as ground truth for symbolic regression tasks.  
- **data_cloud.py** → Script for generating “data clouds”: synthetic samples around each equation, simulating noisy experimental observations.  
- **data_clouds.json** → Serialized data clouds produced by `data_cloud.py`. Contains equation-specific numerical datasets for training.  
- **feynman_parse_trees.json** → JSON representation of Feynman equations as structured parse trees, used for supervised symbolic parsing and concept library training.  
- **predictions.json** → Stores latest model inference results (predicted equations). Used for evaluation against ground truth.  

---

#### `/src/`  
Core implementation, organized into functional modules.  

#### `/parser/`  
Responsible for converting equations into machine-readable structures.  
- **symbolic_parser.py** → Converts string equations (e.g., `"E = mc^2"`) into structured parse trees. Provides alignment between symbolic outputs and dataset trees.  

#### `/embeddings/`  
Encodes symbolic expressions into dense numerical vectors.  
- **t_net_embeddings.py** → T-Net architecture for embedding equations into fixed-length representations.  
- **tnet_embeddings.json** → Precomputed embeddings (older version) emphasizing dataset-wide variation.  
- **tnet_embeddings_new.json** → Improved embeddings with **local feature sensitivity**, optimized for sliding-window sparse decoding.  

#### `/decoder/`  
Implements generative models to decode embeddings into symbolic equations.  
- **decoder.py** → GPT-style autoregressive transformer decoder.  
- **masking_decoder_setup.py** → Utilities for masked sequence modeling and training setup.  
- **sliding_window.py** → Implements memory-efficient **sliding-window sparse attention** for long-sequence decoding.  

#### `/library/`  
Encapsulates reusable symbolic concepts.  
- **learned_library.py** → Maintains a **concept library** of frequently used subtrees (e.g., `sin(x)`, `x^2 + y^2`). Enables compositional reuse during generation.  

#### `/labels/`  
Provides tokenized supervision for training the decoder.  
- **tokenized_gpt_labels.json** → Baseline tokenized labels aligned with dataset.  
- **tokenized_gpt_labels_with_full_funcs.json** → Improved labels including **full functions + concept library tokens**, supporting richer supervision.  

---



## Setup Instructions

To run the pipeline **directly with the latest improvements**:

1.  **Clone the repository and navigate into the project directory:**
    ```bash
    git clone [https://github.com/krishoncloud/SYMBA.git](https://github.com/krishoncloud/SYMBA.git)
    cd SYMBA/SYMBA_REG/SymbolicGPT_Krish_Malik
    ```

2.  **Run the decoder with improved embeddings and labels:**
    ```bash
    python src/decoder/sliding_window.py \
      --embeddings src/embeddings/tnet_embeddings_new.json \
      --labels src/labels/tokenized_gpt_labels_with_full_funcs.json
    ```

---

## Additional Information

### Embeddings

| File | Description | Best For |
| :--- | :--- | :--- |
| `tnet_embeddings_new.json` | New version with **local-focused features** to support sliding-window + sparse decoding. | **Best Performance** |
| `tnet_embeddings.json` | Older version with global/dataset-wide features (useful for broader variation analysis). | Analysis |

### Labels

| File | Description | Best For |
| :--- | :--- | :--- |
| `tokenized_gpt_labels_with_full_funcs.json` | **Improved labels** with learned concept library integration. | **Best Performance** |
| `tokenized_gpt_labels.json` | Baseline labels, without concept library. | Baseline comparison |

**For best performance, use:**
* **Embeddings:** `tnet_embeddings_new.json`
* **Labels:** `tokenized_gpt_labels_with_full_funcs.json`
* **Decoder:** `sliding_window.py`

---

## Recent Additions

- Improved embeddings (`tnet_embeddings_new.json`)
- Improved labels with concept library (`tokenized_gpt_labels_with_full_funcs.json`)
- Sliding-window sparse decoder (`sliding_window.py`)
- Predictions output file (`predictions.json`)
- Medium blog link (`docs/BLOG_LINK.md`)

---

## Blog

Read the detailed writeup here:
[Learning Symbolic Expressions from Data Clouds](https://medium.com/@krishmalikus/learning-symbolic-expressions-from-data-clouds-d186f05435bd)

---

## Credits

ML4SCI / Symba Lab

Developed during **Google Summer of Code (GSoC) 2025** by **@krishoncloud**

Contributions include: restructuring repo, improved embeddings/labels, sliding window decoder, and documentation.
