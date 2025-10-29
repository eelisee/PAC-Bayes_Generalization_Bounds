# Quick Start Guide

This guide will help you run your first PAC-Bayes experiment in 5 minutes.

## Prerequisites

Make sure you have Python 3.8+ installed.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install PyTorch, NumPy, Matplotlib, and other required packages.

## Step 2: Run Your First Experiment

Run a simple experiment with logistic regression on MNIST (0 vs 1 classification):

```bash
python experiments/run_experiment.py
```

This will:
- Download MNIST dataset (if needed)
- Train a logistic regression model
- Record parameter trajectory
- Compute trajectory-averaged Hessian
- Calculate PAC-Bayes bounds
- Compare with baseline methods
- Save results to `results/`

**Expected runtime**: 5-10 minutes on CPU, 2-3 minutes on GPU.

## Step 3: Generate Figures

After the experiment completes, generate figures:

```bash
python experiments/visualize_results.py results/results_*.json figures/
```

This creates:
- `figures/training_curves.png`
- `figures/bound_comparison.png`
- `figures/kl_decomposition.png`
- `figures/results_table.csv`

## Step 4: Explore Interactively

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/interactive_exploration.ipynb
```

This lets you:
- Visualize training progress
- Experiment with hyperparameters
- See KL decomposition in detail
- Compare different configurations

## Common Issues

### Issue: CUDA out of memory
**Solution**: Set `"use_cuda": false` in config file, or reduce batch size.

### Issue: Hessian computation is slow
**Solution**: Use diagonal approximation (`"use_diagonal_hessian": true`) or reduce `"num_hessian_samples"`.

### Issue: PyTorch import error
**Solution**: Make sure PyTorch is installed: `pip install torch torchvision`

## Next Steps

### Experiment with Different Models

Try a small MLP instead of logistic regression:

```bash
python experiments/run_experiment.py --config experiments/config_mlp.json
```

### Tune Hyperparameters

Edit `experiments/config_default.json` to experiment with:
- `c_Q`: Posterior scaling (try 0.1, 1.0, 5.0)
- `lambda_reg`: Regularization (try 0.001, 0.01, 0.1)
- `projection_dim`: Subspace dimension (try 10, 50, 100)
- `epochs`: Training length (try 50, 100, 200)

### Run Multiple Seeds

Increase the number of random seeds for better error estimates:

```json
"seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50]
```

### Custom Experiments

Create your own config file:

```bash
cp experiments/config_default.json experiments/config_custom.json
# Edit config_custom.json
python experiments/run_experiment.py --config experiments/config_custom.json
```

## Understanding the Output

### Terminal Output

You'll see:
```
Running experiment with seed 42
1. Loading data...
   Data loaded: train=10000, val=1500, test=2000, dim=784
2. Creating model...
   Model: logistic, parameters: 785
3. Training model...
   [Progress bar with train/val loss and accuracy]
4. Computing trajectory-averaged Hessian...
   [Progress bar for Hessian estimation]
5. Defining prior...
   Prior: N(0, 1.0 * I)
6. Computing PAC-Bayes bounds...
   Train risk: 0.0234
   Val risk: 0.0256
   Test risk: 0.0267
   
   Trajectory-aware PAC-Bayes bound: 0.0892
     KL divergence: 15.34
     Bound term: 0.0658
   
   Baseline isotropic PAC-Bayes bound: 0.1523
     KL divergence: 392.5
```

### Interpretation

- **Test risk**: True generalization performance (0.0267)
- **Trajectory-aware bound**: Our bound (0.0892)
- **Baseline bound**: Standard isotropic bound (0.1523)

**Gap analysis**: 
- Trajectory-aware gap: 0.0892 - 0.0267 = 0.0625
- Baseline gap: 0.1523 - 0.0267 = 0.1256

Our trajectory-aware bound is **tighter** by ~0.06!

## For the Project Report

After running experiments, you can include:

1. **Figures**: Copy from `figures/` directory
2. **Tables**: Use `figures/results_table.tex` in LaTeX
3. **Results**: Reference values from `results/*.json`

## Getting Help

- Check the main `README.md` for detailed documentation
- See `CODE_ATTRIBUTION.md` for implementation details
- Open an issue on GitHub for bugs
- Contact the authors for questions

## Summary Commands

```bash
# Complete workflow
pip install -r requirements.txt
python experiments/run_experiment.py
python experiments/visualize_results.py results/results_*.json figures/
jupyter notebook notebooks/interactive_exploration.ipynb
```

That's it! You're ready to explore PAC-Bayes generalization bounds.
