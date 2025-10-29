# Trajectory-Aware PAC-Bayes Generalization Bounds

Repository for the final project of the Advanced Machine Learning course (taught by Austin Stromme during the final year at ENSAE Paris).

## Project Overview

This project implements **trajectory-aware PAC-Bayes bounds** for high-dimensional models, with explicit Gaussian KL decomposition and stochastic error control. We estimate generalization bounds using trajectory-averaged Hessian information during optimization.

### Key Contributions

1. **Trajectory-averaged Hessian estimation** along SGD optimization paths
2. **Gaussian KL decomposition** into top-subspace alignment, residual complexity, and curvature terms
3. **Stochastic trace and log-determinant estimation** with Hutchinson estimator
4. **Empirical validation** on MNIST binary classification with multiple model architectures

## Repository Structure

```
PAC-Bayes_Generalization_Bounds/
├── src/                          # Core implementation
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── models.py                # Model architectures (logistic, MLP)
│   ├── training.py              # Training with trajectory recording
│   ├── hessian_estimator.py    # Hessian computation and averaging
│   ├── pac_bayes_bounds.py      # PAC-Bayes bound computation
│   └── utils.py                 # Utility functions (Hutchinson, etc.)
│
├── experiments/                  # Experiment scripts
│   ├── run_experiment.py        # Main experiment runner
│   ├── visualize_results.py     # Visualization and plotting
│   ├── config_default.json      # Default configuration (logistic)
│   └── config_mlp.json          # MLP configuration
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── (will contain analysis notebooks)
│
├── results/                      # Experiment results (JSON)
├── figures/                      # Generated figures for report
├── data/                         # Dataset storage (MNIST)
│
├── paper.tex                     # Technical paper
├── implementation.tex            # Implementation plan
├── projectreport.tex            # Project report (for submission)
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/eelisee/PAC-Bayes_Generalization_Bounds.git
cd PAC-Bayes_Generalization_Bounds
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

1. **Basic experiment with default configuration (logistic regression)**:
```bash
python experiments/run_experiment.py
```

2. **Run with custom configuration**:
```bash
python experiments/run_experiment.py --config experiments/config_mlp.json --output results/mlp_experiment
```

3. **Configuration parameters** (see `experiments/config_default.json`):
   - `c_Q`: Posterior scaling constant (default: 1.0)
   - `lambda_reg`: Regularization parameter λ (default: 0.01)
   - `projection_dim`: Top-subspace dimension r (default: 10)
   - `num_hutchinson_samples`: Number of Hutchinson samples m (default: 100)
   - `epochs`: Training epochs (default: 100)
   - `seeds`: Random seeds for reproducibility

### Generating Figures

After running experiments, generate figures for the report:

```bash
python experiments/visualize_results.py results/results_<timestamp>.json figures/
```

This will generate:
- `training_curves.png`: Training and validation curves with error bars
- `bound_comparison.png`: Comparison of bounds vs empirical risk
- `kl_decomposition.png`: KL divergence decomposition
- `results_table.csv` / `results_table.tex`: Summary table

### Example Workflow

```bash
# 1. Run experiment with logistic regression
python experiments/run_experiment.py --config experiments/config_default.json --output results/logistic

# 2. Run experiment with MLP
python experiments/run_experiment.py --config experiments/config_mlp.json --output results/mlp

# 3. Generate figures for logistic regression results
python experiments/visualize_results.py results/logistic/results_*.json figures/logistic/

# 4. Generate figures for MLP results
python experiments/visualize_results.py results/mlp/results_*.json figures/mlp/
```

## Implementation Details

### Original vs. Adapted Code

**Original Implementation (Novel Contributions):**
- Trajectory recording mechanism during SGD optimization (`training.py`)
- Trajectory-averaged Hessian computation (`hessian_estimator.py`)
- Gaussian KL decomposition with top-subspace projection (`pac_bayes_bounds.py`)
- Stochastic trace and log-determinant estimation (`utils.py`, `pac_bayes_bounds.py`)
- Integration of these components into unified PAC-Bayes framework

**Adapted from Standard Methods:**
- PyTorch training loop structure (adapted with trajectory recording)
- MNIST data loading from torchvision
- Standard Hutchinson trace estimator (extended for log-determinant)
- Standard PyTorch autograd for Hessian computation

All adaptations are clearly documented in code comments.

## Reproducibility

### Random Seeds
All experiments use fixed random seeds (default: [42, 43, 44, 45, 46]) for reproducibility. Seeds are set for:
- NumPy random number generator
- PyTorch CPU and GPU random number generators
- PyTorch cuDNN (deterministic mode)

### Dataset
MNIST dataset is automatically downloaded on first run and cached in `./data/`.

### Hardware
Experiments can run on CPU or GPU (CUDA). GPU is automatically detected and used if available.

## Theoretical Background

See `paper.tex` for detailed theoretical foundations, including:
- PAC-Bayes preliminaries and conditional PAC-Bayes bounds
- Gaussian posterior construction with trajectory-averaged Hessian
- KL decomposition into top-subspace, residual, and curvature terms
- Stochastic error control with dimensional dependence
- Rigorous treatment of assumptions and limitations

## Results

Results are saved in JSON format with the following structure:
```json
{
  "seed": 42,
  "train_risk": 0.0234,
  "test_risk": 0.0267,
  "trajectory_bound": {
    "pac_bayes_bound": 0.0892,
    "kl_divergence": 15.34,
    "bound_term": 0.0658
  },
  "baseline_bound": {
    "pac_bayes_bound": 0.1523
  },
  "kl_decomposition": { ... },
  "training_history": { ... }
}
```

Multiple seeds are averaged with standard deviations reported as error bars.

## Citation

If you use this code, please cite:

```
@misc{pacbayes_trajectory_2025,
  title={Trajectory-Aware PAC-Bayes Bounds with High-Dimensional Stochastic Error Control},
  author={Elise Wolf},
  year={2025},
  institution={ENSAE Paris}
}
```

## License

This project is part of academic coursework at ENSAE Paris.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

- Course instructor: Austin Stromme
- Institution: ENSAE Paris
- Course: Advanced Machine Learning (Final Year)
