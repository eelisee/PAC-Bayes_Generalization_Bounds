"""
Main experiment runner for PAC-Bayes bounds experiments.
========================================================

This script orchestrates the complete experimental pipeline:
1. Load and preprocess data
2. Train models with trajectory recording
3. Compute trajectory-averaged Hessians
4. Compute PAC-Bayes bounds with KL decomposition
5. Compare with baseline methods
6. Generate plots and tables

Run with: python experiments/run_experiment.py --config experiments/config.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_mnist_binary
from src.models import create_model, count_parameters
from src.training import train_model_with_trajectory
from src.hessian_estimator import TrajectoryHessianEstimator
from src.pac_bayes_bounds import (
    GaussianPACBayesBound,
    TrajectoryCurvatureTerm,
    compute_baseline_pac_bayes_bound
)
from src.utils import set_random_seed

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def run_single_experiment(config: dict, seed: int) -> dict:
    """
    Run a single experiment with given configuration and seed.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
    seed : int
        Random seed
        
    Returns:
    --------
    results : dict
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with seed {seed}")
    print(f"{'='*60}\n")
    
    # Set random seed
    set_random_seed(seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader, input_dim = get_mnist_binary(
        data_dir=config['data_dir'],
        class_0=config['class_0'],
        class_1=config['class_1'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        flatten=config['flatten'],
        seed=seed
    )
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Data loaded: train={n_train}, val={n_val}, test={n_test}, dim={input_dim}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        **config.get('model_kwargs', {})
    )
    n_params = count_parameters(model)
    print(f"Model: {config['model_type']}, parameters: {n_params}")
    
    # Train with trajectory recording
    print("\n3. Training model with trajectory recording...")
    model, history = train_model_with_trajectory(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        optimizer_type=config['optimizer'],
        device=device,
        record_every=config['record_every'],
        seed=seed,
        verbose=True
    )
    
    print(f"Training complete. Trajectory length: {len(history['trajectory'])}")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"Final train acc: {history['train_accs'][-1]:.4f}")
    if history['val_losses']:
        print(f"Final val loss: {history['val_losses'][-1]:.4f}")
        print(f"Final val acc: {history['val_accs'][-1]:.4f}")
    
    # Compute trajectory-averaged Hessian and posterior
    print("\n4. Computing trajectory-averaged Hessian...")
    hessian_estimator = TrajectoryHessianEstimator(
        model=model,
        trajectory=history['trajectory'],
        data_loader=train_loader,
        criterion=torch.nn.BCELoss(),
        device=device,
        lambda_reg=config['lambda_reg'],
        c_Q=config['c_Q'],
        use_diagonal=config['use_diagonal_hessian'],
        num_hess_samples=config['num_hessian_samples'],
        verbose=True
    )
    
    mu_Q, Sigma_Q = hessian_estimator.get_posterior_params()
    print(f"Posterior computed: mu_Q shape={mu_Q.shape}, Sigma_Q shape={Sigma_Q.shape}")
    
    # Define prior (isotropic Gaussian)
    print("\n5. Defining prior...")
    prior_variance = config['prior_variance']
    if config['use_diagonal_hessian']:
        mu_P = torch.zeros_like(mu_Q)
        Sigma_P = torch.ones_like(Sigma_Q) * prior_variance
    else:
        mu_P = torch.zeros_like(mu_Q)
        Sigma_P = torch.eye(mu_Q.shape[0], device=device) * prior_variance
    
    print(f"Prior: N(0, {prior_variance} * I)")
    
    # Compute PAC-Bayes bound
    print("\n6. Computing PAC-Bayes bounds...")
    
    # Compute empirical risks
    from src.training import TrajectoryTrainer
    trainer = TrajectoryTrainer(
        model=model,
        optimizer=None,
        criterion=torch.nn.BCELoss(),
        device=device,
        verbose=False
    )
    
    train_risk = trainer.compute_empirical_risk(train_loader)
    val_risk = trainer.compute_empirical_risk(val_loader)
    test_risk = trainer.compute_empirical_risk(test_loader)
    
    print(f"Empirical risks: train={train_risk:.4f}, val={val_risk:.4f}, test={test_risk:.4f}")
    
    # Our trajectory-aware bound
    pac_bayes_computer = GaussianPACBayesBound(
        mu_P=mu_P,
        Sigma_P=Sigma_P,
        mu_Q=mu_Q,
        Sigma_Q=Sigma_Q,
        r=config['projection_dim'],
        device=device,
        use_diagonal=config['use_diagonal_hessian']
    )
    
    kl_decomposition = pac_bayes_computer.compute_kl_decomposition(
        num_hutchinson_samples=config['num_hutchinson_samples']
    )
    
    bound_info = pac_bayes_computer.compute_pac_bayes_bound(
        empirical_risk=train_risk,
        n_samples=n_train,
        delta=config['delta'],
        epsilon_stochastic=config.get('epsilon_stochastic', 0.0)
    )
    
    print(f"\nTrajectory-aware PAC-Bayes bound: {bound_info['pac_bayes_bound']:.4f}")
    print(f"  KL divergence: {bound_info['kl_divergence']:.4f}")
    print(f"  Bound term: {bound_info['bound_term']:.4f}")
    
    # Baseline bound
    baseline_bound_info = compute_baseline_pac_bayes_bound(
        empirical_risk=train_risk,
        n_samples=n_train,
        d=n_params,
        delta=config['delta'],
        prior_variance=prior_variance,
        posterior_variance=config.get('baseline_posterior_variance', 0.01)
    )
    
    print(f"\nBaseline isotropic PAC-Bayes bound: {baseline_bound_info['pac_bayes_bound']:.4f}")
    print(f"  KL divergence: {baseline_bound_info['kl_divergence']:.4f}")
    
    # Package results
    results = {
        'seed': seed,
        'config': config,
        'n_params': n_params,
        'n_train': n_train,
        'train_risk': train_risk,
        'val_risk': val_risk,
        'test_risk': test_risk,
        'trajectory_length': len(history['trajectory']),
        'kl_decomposition': kl_decomposition,
        'trajectory_bound': bound_info,
        'baseline_bound': baseline_bound_info,
        'training_history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'train_accs': history['train_accs'],
            'val_accs': history['val_accs']
        }
    }
    
    return results


def run_experiments(config: dict) -> list:
    """
    Run experiments with multiple random seeds.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
        
    Returns:
    --------
    all_results : list
        List of results for each seed
    """
    all_results = []
    
    for seed in config['seeds']:
        try:
            results = run_single_experiment(config, seed)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in experiment with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


def save_results(results: list, output_dir: str):
    """Save experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    results_file = output_path / f"results_{timestamp}.json"
    
    # Convert numpy/torch types to native Python for JSON
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_clean = convert_types(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PAC-Bayes experiments")
    parser.add_argument('--config', type=str, default='experiments/config_default.json',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = get_default_config()
    
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Run experiments
    print("\n" + "="*60)
    print("STARTING EXPERIMENTS")
    print("="*60)
    
    results = run_experiments(config)
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        test_risks = [r['test_risk'] for r in results]
        traj_bounds = [r['trajectory_bound']['pac_bayes_bound'] for r in results]
        base_bounds = [r['baseline_bound']['pac_bayes_bound'] for r in results]
        
        print(f"\nTest risk: {np.mean(test_risks):.4f} ± {np.std(test_risks):.4f}")
        print(f"Trajectory-aware bound: {np.mean(traj_bounds):.4f} ± {np.std(traj_bounds):.4f}")
        print(f"Baseline bound: {np.mean(base_bounds):.4f} ± {np.std(base_bounds):.4f}")


def get_default_config():
    """Get default configuration."""
    return {
        'data_dir': './data',
        'class_0': 0,
        'class_1': 1,
        'batch_size': 128,
        'val_split': 0.15,
        'flatten': True,
        'model_type': 'logistic',
        'model_kwargs': {},
        'epochs': 50,
        'learning_rate': 0.01,
        'optimizer': 'sgd',
        'record_every': 5,
        'lambda_reg': 0.01,
        'c_Q': 1.0,
        'use_diagonal_hessian': True,
        'num_hessian_samples': 50,
        'projection_dim': 10,
        'num_hutchinson_samples': 100,
        'prior_variance': 1.0,
        'baseline_posterior_variance': 0.01,
        'delta': 0.05,
        'epsilon_stochastic': 0.0,
        'seeds': [42, 43, 44],
        'use_cuda': True
    }


if __name__ == "__main__":
    main()
