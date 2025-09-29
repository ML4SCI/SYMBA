# Foundational Model for High Energy Physics - Symbolic Regression
### Google Summer of Code 2025 - Final Work Product

---

**Organizations:** Google School of Code & University of Alabama

**Student:** Miche Maral

**Mentors:**
- Sergei Gleyzer (University of Alabama)
- Nobuchika Okada (University of Alabama)
- Eric Reinhardt (University of Alabama)

Project Overview
This project develops a foundational model for symbolic regression in High Energy Physics, aiming to decipher complex physical equations. By leveraging modern sequence-to-sequence architectures like the Transformer and Mamba, it addresses the critical need for interpretable AI in scientific discovery. The work successfully demonstrates the viability of a Transformer-based approach, achieving near-perfect accuracy in generating symbolic expressions and setting the stage for direct comparison with emerging state-space models.

Results and Performance
The Transformer model was trained for 15 epochs and demonstrated exceptional performance. The model converged rapidly after the first epoch, achieving near-perfect token and sequence-level accuracy on the validation set. This indicates a strong capacity for learning the underlying patterns of the symbolic expressions.

The final training results are summarized below:

| Epoch | Train Loss | Val Loss | Val Token Acc | Val Seq Acc |
| :---: | :--------: | :------: | :-----------: | :---------: |
| 1     | 1.1326     | 0.4319   | 0.8396        | 0.0000      |
| 2     | 0.2125     | 0.0001   | 1.0000        | 0.9797      |
| 3     | 0.0031     | 0.0000   | 1.0000        | 0.9884      |
| 4     | 0.0014     | 0.0000   | 1.0000        | 0.9884      |
| 5     | 0.0010     | 0.0000   | 1.0000        | 0.9884      |
| ...   | ...        | ...      | ...           | ...         |
| 14    | 0.0001     | 0.0000   | 1.0000        | 0.9971      |
| 15    | 0.0001     | 0.0000   | 1.0000        | 0.9971      |

Deliverables
Core Implementations: A PyTorch-based Transformer model and a Mamba SSM model architecture for symbolic regression, contained within a single comprehensive notebook.

Documentation and Usage Examples: The main Jupyter Notebook (process_train_mamba__and_transformer.ipynb) serves as a complete example, covering data processing, model training, and validation.

Tests and Validation: An integrated validation loop within the training script measures model performance using token and sequence-level accuracy.

Final Project Write-Up: A detailed report outlining the project's goals, methods, and outcomes.

Final Report
A detailed explanation of the project's goals, design, implementation, and results can be found in the final write-up on Medium: [Link to Medium Post]

Installation and Usage
To get started, install the necessary dependencies:

!pip install torch pandas tqdm mamba-ssm notebook

After installation, you can run the process_train_mamba__and_transformer.ipynb notebook in a Jupyter environment to replicate the data processing and model training workflows.

## Repository Structure

```text
.
├── process_train_mamba__and_transformer.ipynb   # Main notebook with all code
├── data/                                        # Directory for datasets (not included)
│   ├── normalized_expressions_output_train.csv
│   └── normalized_expressions_output_valid.csv
└── readme.md                                    # This file
```
Future Work
This project has established a strong baseline and can be extended in several key areas:

Mamba Model Training and Evaluation: The Mamba architecture is implemented but requires training and a full comparative analysis against the Transformer's performance on metrics like accuracy, speed, and memory usage.

Inference and Application: Develop a dedicated inference script to allow the model to predict expressions from new inputs, turning it into a practical tool for physicists.

Hyperparameter Optimization and Scaling: Conduct a systematic search for optimal hyperparameters and explore scaling the models with a larger, more complex dataset to enhance their foundational capabilities.

Acknowledgements
Thank you to my mentors and the Google Summer of Code organization for their guidance and support throughout this project.
