This Repository is part of SYMBA Project under GSoC 2025 Program, This Repository contains Implementations of two architectures
1. SymKAN-TP-Transformer
2. SineKAN-MoE

Find the Implementation of SineKAN-MoE in the `./src` directory or directly click this link **<a href = "https://github.com/Ayushmishra05/SymKAN-TP-Transformer/tree/main/src/SineKAN_MoE" style = "color:blue"> SineKAN-MoE </a>** 
# SymKAN-TP-Transformer (Tensor-Product Transformer combined with SineKAN)

## Project Structure 
```
.
├── LICENSE
├── README.md
├── notebooks
│   ├── preprocess.ipynb
│   ├── sym_kan_transformer.ipynb
│   ├── tokenizer.ipynb
│   └── vanilla_transformer.ipynb
├── preprocess
│   ├── __init__.py
│   └── tokenizersplit.py
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── sym_kan_transformer
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── build_model.cpython-312.pyc
│   │   │   ├── config.cpython-312.pyc
│   │   │   ├── model.cpython-312.pyc
│   │   │   ├── tokenizer.cpython-312.pyc
│   │   │   └── train.cpython-312.pyc
│   │   ├── build_model.py
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── tokenizer.py
│   │   └── train.py
│   └── transformer
│       ├── train.py
│       └── vanilla_transformer.py
└── train_sym_kan_transformer.py

```

## Task 1 :  Extracting the Data From the sources 
Starting off with the Data Extraction part, The Data was collected from the listed source, the data was in the Raw Text format, the first task was to convert this Raw text data into a CSV formatted dataset
* I have used re (python module), which is python module for regular expressions 
* The Dataset was converted into individual variables, and then was embedded into the dataset 
* Vertex was not consistent across the data, this might be the case because, some of the particle collisions doesn't happen between vertices but it only depends on the single vertex, that's why the case 


## Task 2 : Tokenization  
The Tokenization technique used here was 
   **Mathematical Aware Tokenization Method** 
   However there were various tokenization methods, that was explored, like **Character Wise Encoding, Byte Pair Encoding**, out of this **Mathematical Aware tokenization gave the good results**, The Working of this Tokenization Method is explained in this Paper <a href="https://cdn.iiit.ac.in/cdn/web2py.iiit.ac.in/research_centres/publications/download/inproceedings.pdf.867521e9a9170b72.312e393738313631313937373137322e33332e706466.pdf" style="color:blue;">Paper Link</a> (This paper shows, ho Performance increases significantly in math and physics related tasks, using this tokenization method). 


## Task 3 : Transformer 
the Transformer model was trained on the dataset, initially the plan was to use the Decoder Architecture, with the SineKAN layers, but because Decoders only works in language tasks, i cancelled the plan of training Decoder 

* This Transformer architecture is the same as, which is provided in the paper, Now for comparison a basic Transformer was built to train on the dataset, the dataset consisted of 15K Rows, which was provided, by the org
* After training the Transformers for 10 epochs, the Accuracy came out to be 99.5%, Now the task was to improve it and also bring the interpretability here 

## Task 3.1 - Approach - SymKAN-TPT (Tensor-Product Transformer combined with KAN and Sympy)
 ### Why this Approach 
* The Last Approach for the Same Problem Introduced a SineKAN Layer with Transformer Architecture, which resulted in Promising Accuracy 
* This Approach is Inspired from the Last Approach, We utilise SineKAN Layer with TP-Transformer Further Evaluated by Sympy Layer 
  
  ### Why Choose TP-Transformer ? 
   * Unlike the **standard Transformer**, which simply adds token embeddings and positional encodings, the TP-Transformer uses a **tensor product representation (TPR)**. It multiplies a token embedding (representing the "what") with a role vector (representing the "where"), creating a richer representation of each token’s content and its position or role in the sequence.

   * Due to their ability of Creating a **richer representation** for each toke, they are preferred in **Symbolic Reasoning Tasks**

   * The **Role Based Vector** multiplied with **attention-weighted** values makes the model better at capturing position-sentsitive dependencies which makes it reusable for tasks like **parsing and expression generation** 

   * The TP-Transformer is inspired by **<a href = "https://arxiv.org/pdf/1910.06611" style = "color:blue"> Enhancing the Transformer With Explicit Relational
Encoding for Math Problem Solving </a>** 

  ### Integration of SineKAN Layer (Sinusoidal Kolmogorov Arnold Networks)

  * This has been observed that, KAN Layer are proficient at Symbolic Tasks, and also at capturing the non-linear and complex patterns in the data. 
  * However due to the complexity and computational cost asscoiated it KAN, there are multiple alternative of KAN that promises better results than KAN, one such implementation is SineKAN

  * This Motivated me to integrate it in the architecture, the last Softmax layer from the Transformer was replaced by the SineKAN Layer.

  * This helped the model to generate and understand the rule-based sequences 


  ### Model Efficiency 
  * The TP-Transformer ties the weights of its input embedding layer to its output projection layer, reusing the same parameters for both. While some standard Transformer implementations do this too, it’s a core feature here that complements the model’s design. 

  * This reduces the total number of parameters, making the model more memory-efficient and potentially improving generalization. 

  ### Sympy Layer Integration 
  * Sympy (Symbolic Python) is a python module, which is used for symbolic Calculations 

  * A key role of the SymPy layer is to assess the correctness of the TP-Transformer’s predictions at a symbolic level. It uses functions like compare_expressions to determine if two mathematical expressions are equivalent, even if they appear different (E.g  : **(x^2 - 1) = (x -1) (x + 1)** ) 
  
  * This symbolic comparison is far more robust than token-level matching, as it verifies the underlying mathematical structure rather than just surface-level syntax. this ensures the model’s outputs are not only correct but also meaningful. 


 

## What about the Complexity ?

   * The SymKan-Tp-Transformer uses more computation than the traditional transformer architecture, The Reasons are below 

      * **Reasons** : 
      * The standard Transformer’s embedding is lightweight at **𝑂(𝑁)**, while the TP-Transformer’s **𝑂(𝑁⋅𝑑𝑥^2)** scales with the square of the hidden dimension, making it far more computationally demanding, especially for large **𝑑𝑥** 

      * Both models share the 𝑂(𝑁^2⋅𝑑𝑥) attention bottleneck, but the TP-Transformer’s additional role-related operations increase the constant factors, making each layer slower.

      *  Overall The Memory Requirements of Tp-Transformer is More compared to Standard Transformer, but this is often considered as a **Trade-off** between the **Memory and the Performance** 

## Project Status : (Under Development)
