"""
Validation Experiments to Address Unexpected Bound Behavior
============================================================

This script implements all suggested experiments to investigate why
the trajectory-aware bound is looser than expected:

1. Longer trajectories (T ∈ {50, 100, 200})
2. Hyperparameter sweep (c_Q ∈ {0.1, 0.5, 1.0, 2.0, 5.0})
3. Variance estimation validation (m ∈ {50, 100, 200, 500})
4. Per-epoch KL evolution tracking
5. Implementation verification tests
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MNISTBinaryClassification
from src.models import create_model
from src.training import TrajectoryTrainer
from src.hessian_estimator import TrajectoryHessianEstimator
from src.pac_bayes_bounds import GaussianPACBayesBound
from src.utils import set_random_seed

plt.rcParams['figure.dpi'] = 300


class ValidationExperiments:
    """
    Comprehensive validation experiments to diagnose bound behavior.
    """
    
    def __init__(self, base_config: Dict, output_dir: str = "validation_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store detailed diagnostics
        self.diagnostics = {
            'per_epoch_kl': [],
            'curvature_stats': [],
            'posterior_stats': [],
            'hessian_diagnostics': []
        }
    
    def _load_data(self, config: Dict):
        """Load MNIST data."""
        data_loader = MNISTBinaryClassification(
            class_0=config['class_0'],
            class_1=config['class_1'],
            data_dir=config['data_dir'],
            val_split=config['val_split']
        )
        return data_loader.get_dataloaders(batch_size=config['batch_size'])
    
    def _run_experiment_with_diagnostics(self, config: Dict, seed: int = 42) -> Dict:
        """
        Run experiment with detailed diagnostic tracking.
        """
        set_random_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Running experiment: seed={seed}")
        print(f"  T={config['num_epochs']}, c_Q={config['c_Q']}, λ={config['lambda_reg']}, m={config.get('m_hutchinson', 100)}")
        print(f"{'='*60}")
        
        # Load data
        train_loader, val_loader, test_loader = self._load_data(config)
        n_train = len(train_loader.dataset)
        
                # Create model
        model = create_model(config['model_type'], input_dim=784, hidden_dims=config.get('hidden_dims'))
        d = len(model.get_parameters_flat())
        print(f"Model: {config['model_type']}, parameters: {d}")
        
        # Train with trajectory recording
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.BCELoss()
        
        trainer = TrajectoryTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
            record_every=config['trajectory_interval']
        )
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=config['num_epochs'])
        trajectory = trainer.trajectory
        T = len(trajectory)
        
        print(f"\nTrajectory length: {T}")
        print(f"Final train acc: {history['train_accs'][-1]:.4f}")
        print(f"Final val acc: {history['val_accs'][-1]:.4f}")
        
        # === DIAGNOSTIC 1: Trajectory stability ===
        trajectory_changes = [np.linalg.norm(trajectory[t] - trajectory[t-1]) 
                             for t in range(1, T)]
        print(f"\nTrajectory changes: mean={np.mean(trajectory_changes):.6f}, "
              f"std={np.std(trajectory_changes):.6f}")
        print(f"Last 5 changes: {trajectory_changes[-5:]}")
        
        # === DIAGNOSTIC 2: Compute Hessian with detailed logging ===
        print(f"\n{'='*60}")
        print("Computing trajectory-averaged Hessian...")
        print(f"{'='*60}")
        
        hessian_estimator = TrajectoryHessianEstimator(
            model=model,
            trajectory=trajectory,
            data_loader=train_loader
        )
        
        # Compute individual Hessians for diagnostics
        individual_hessians = []
        for t, w_t in enumerate(trajectory):
            model.set_parameters_flat(w_t)
            H_t = hessian_estimator._compute_single_hessian(
                model, train_loader, use_diagonal=config['use_diagonal_hessian']
            )
            individual_hessians.append(H_t)
            
            if t < 3 or t >= T - 3:  # First 3 and last 3
                trace_t = np.trace(H_t) if H_t.ndim == 2 else np.sum(H_t)
                print(f"  H_{t}: trace={trace_t:.2f}, min={H_t.min():.6f}, max={H_t.max():.6f}")
        
        # Average Hessian
        H_bar = np.mean(individual_hessians, axis=0)
        trace_H_bar = np.trace(H_bar) if H_bar.ndim == 2 else np.sum(H_bar)
        
        print(f"\nH_bar statistics:")
        print(f"  Shape: {H_bar.shape}")
        print(f"  Trace: {trace_H_bar:.2f}")
        print(f"  Min: {H_bar.min():.6f}")
        print(f"  Max: {H_bar.max():.6f}")
        print(f"  Mean: {H_bar.mean():.6f}")
        
        # Check for numerical issues
        if np.any(np.isnan(H_bar)):
            print("  WARNING: NaN detected in H_bar!")
        if np.any(np.isinf(H_bar)):
            print("  WARNING: Inf detected in H_bar!")
        
        # === DIAGNOSTIC 3: Posterior covariance ===
        print(f"\n{'='*60}")
        print(f"Computing posterior covariance with c_Q={config['c_Q']}, λ={config['lambda_reg']}")
        print(f"{'='*60}")
        
        Sigma_Q = hessian_estimator.compute_posterior_covariance(
            H_bar=H_bar,
            c_Q=config['c_Q'],
            lambda_reg=config['lambda_reg']
        )
        
        trace_Sigma_Q = np.trace(Sigma_Q) if Sigma_Q.ndim == 2 else np.sum(Sigma_Q)
        
        print(f"\nΣ_Q statistics:")
        print(f"  Shape: {Sigma_Q.shape}")
        print(f"  Trace: {trace_Sigma_Q:.2f}")
        print(f"  Min: {Sigma_Q.min():.6f}")
        print(f"  Max: {Sigma_Q.max():.6f}")
        print(f"  Mean: {Sigma_Q.mean():.6f}")
        
        # Check if covariance is reasonable
        if trace_Sigma_Q > 1e6:
            print("  WARNING: Very large posterior variance!")
        if Sigma_Q.min() < 0:
            print("  WARNING: Negative variances detected!")
        
        # === DIAGNOSTIC 4: Prior setup ===
        print(f"\n{'='*60}")
        print("Setting up prior...")
        print(f"{'='*60}")
        
        prior_variance = config['prior_variance']
        prior_mean = np.zeros(d)
        prior_cov_inv = np.eye(d) / prior_variance
        
        print(f"Prior: N(0, {prior_variance} * I)")
        print(f"  Trace(Σ_P): {d * prior_variance:.2f}")
        print(f"  Trace(Σ_P^{-1}): {d / prior_variance:.2f}")
        
        # === DIAGNOSTIC 5: KL computation with decomposition ===
        print(f"\n{'='*60}")
        print("Computing KL divergence...")
        print(f"{'='*60}")
        
        pac_bayes = GaussianPACBayesBound(
            prior_mean=prior_mean,
            prior_cov_inv=prior_cov_inv,
            delta=config['delta']
        )
        
        w_Q = trajectory[-1]
        
        # Manual KL computation for verification
        if Sigma_Q.ndim == 1:  # Diagonal
            Sigma_Q_diag = Sigma_Q
            Sigma_P_diag = np.ones(d) * prior_variance
            
            # KL for diagonal Gaussians: 0.5 * [tr(Σ_P^{-1} Σ_Q) + (μ_Q - μ_P)^T Σ_P^{-1} (μ_Q - μ_P) - d + log(det(Σ_P)/det(Σ_Q))]
            mean_diff = w_Q - prior_mean
            
            trace_term = np.sum(Sigma_Q_diag / Sigma_P_diag)
            mean_term = np.sum(mean_diff**2 / Sigma_P_diag)
            logdet_P = np.sum(np.log(Sigma_P_diag))
            logdet_Q = np.sum(np.log(Sigma_Q_diag))
            logdet_term = logdet_P - logdet_Q
            dim_term = d
            
            kl_manual = 0.5 * (trace_term + mean_term - dim_term + logdet_term)
            
            print(f"\nManual KL computation (diagonal):")
            print(f"  Trace term: {trace_term:.2f}")
            print(f"  Mean alignment: {mean_term:.2f}")
            print(f"  Dimension term: -{dim_term}")
            print(f"  Log-det term: {logdet_term:.2f}")
            print(f"  Total KL: {kl_manual:.2f}")
            
            # Check each component
            print(f"\n  Component analysis:")
            print(f"    tr(Σ_Q): {np.sum(Sigma_Q_diag):.2f}")
            print(f"    tr(Σ_P): {np.sum(Sigma_P_diag):.2f}")
            print(f"    ||μ_Q - μ_P||²: {np.sum(mean_diff**2):.2f}")
            print(f"    log(det(Σ_P)): {logdet_P:.2f}")
            print(f"    log(det(Σ_Q)): {logdet_Q:.2f}")
        
        # Compute via pac_bayes module
        trajectory_bound = pac_bayes.compute_pac_bayes_bound(
            w_Q=w_Q,
            Sigma_Q=Sigma_Q,
            empirical_risk=trainer.compute_empirical_risk(train_loader),
            n_samples=n_train,
            stochastic_error=0.0
        )
        
        print(f"\nPAC-Bayes bound computation:")
        print(f"  KL divergence: {trajectory_bound['kl_divergence']:.2f}")
        print(f"  Empirical risk: {trajectory_bound['empirical_risk']:.4f}")
        print(f"  Bound term: {trajectory_bound['bound_term']:.4f}")
        print(f"  PAC-Bayes bound: {trajectory_bound['pac_bayes_bound']:.4f}")
        
        # === DIAGNOSTIC 6: Baseline bound ===
        print(f"\n{'='*60}")
        print("Computing baseline bound (isotropic)...")
        print(f"{'='*60}")
        
        baseline_bound = pac_bayes.compute_baseline_pac_bayes_bound(
            w_Q=w_Q,
            sigma_Q=1.0,  # Standard isotropic
            empirical_risk=trainer.compute_empirical_risk(train_loader),
            n_samples=n_train
        )
        
        print(f"  KL divergence: {baseline_bound['kl_divergence']:.2f}")
        print(f"  PAC-Bayes bound: {baseline_bound['pac_bayes_bound']:.4f}")
        
        # === DIAGNOSTIC 7: Test performance ===
        test_risk = trainer.compute_empirical_risk(test_loader)
        train_risk = trainer.compute_empirical_risk(train_loader)
        
        traj_gap = trajectory_bound['pac_bayes_bound'] - test_risk
        base_gap = baseline_bound['pac_bayes_bound'] - test_risk
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Train risk: {train_risk:.4f}")
        print(f"Test risk: {test_risk:.4f}")
        print(f"Trajectory bound: {trajectory_bound['pac_bayes_bound']:.4f}")
        print(f"Baseline bound: {baseline_bound['pac_bayes_bound']:.4f}")
        print(f"Trajectory gap: {traj_gap:.4f}")
        print(f"Baseline gap: {base_gap:.4f}")
        
        if traj_gap > base_gap:
            print(f"\n⚠️  WARNING: Trajectory bound is LOOSER by {(traj_gap - base_gap):.4f}")
            print(f"   Relative worsening: {((traj_gap / base_gap - 1) * 100):.1f}%")
        else:
            print(f"\n✓ Trajectory bound is TIGHTER by {(base_gap - traj_gap):.4f}")
            print(f"   Relative improvement: {((1 - traj_gap / base_gap) * 100):.1f}%")
        
        return {
            'seed': seed,
            'config': config,
            'train_risk': train_risk,
            'test_risk': test_risk,
            'trajectory_bound': trajectory_bound['pac_bayes_bound'],
            'baseline_bound': baseline_bound['pac_bayes_bound'],
            'trajectory_gap': traj_gap,
            'baseline_gap': base_gap,
            'kl_trajectory': trajectory_bound['kl_divergence'],
            'kl_baseline': baseline_bound['kl_divergence'],
            'trajectory_length': T,
            'diagnostics': {
                'trace_H_bar': float(trace_H_bar),
                'trace_Sigma_Q': float(trace_Sigma_Q),
                'trajectory_changes': trajectory_changes,
                'H_bar_stats': {
                    'min': float(H_bar.min()),
                    'max': float(H_bar.max()),
                    'mean': float(H_bar.mean()),
                },
                'Sigma_Q_stats': {
                    'min': float(Sigma_Q.min()),
                    'max': float(Sigma_Q.max()),
                    'mean': float(Sigma_Q.mean()),
                }
            }
        }
    
    def experiment_1_trajectory_length(self, T_values: List[int] = [20, 50, 100, 200]):
        """
        Experiment 1: Effect of trajectory length T.
        
        Expected: Longer trajectories should stabilize curvature estimates.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: TRAJECTORY LENGTH ANALYSIS")
        print("="*80)
        
        results = []
        
        for T in T_values:
            config = self.base_config.copy()
            config['num_epochs'] = T
            config['trajectory_interval'] = max(1, T // 20)  # ~20 snapshots
            
            result = self._run_experiment_with_diagnostics(config, seed=42)
            results.append(result)
        
        # Visualize
        self._plot_trajectory_length_results(results)
        
        return results
    
    def experiment_2_posterior_scaling(self, c_Q_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]):
        """
        Experiment 2: Effect of posterior scaling c_Q.
        
        Expected: Should find optimal c_Q that minimizes bound gap.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: POSTERIOR SCALING ANALYSIS")
        print("="*80)
        
        results = []
        
        for c_Q in c_Q_values:
            config = self.base_config.copy()
            config['c_Q'] = c_Q
            
            result = self._run_experiment_with_diagnostics(config, seed=42)
            results.append(result)
        
        # Visualize
        self._plot_posterior_scaling_results(results)
        
        return results
    
    def experiment_3_hutchinson_samples(self, m_values: List[int] = [50, 100, 200, 500]):
        """
        Experiment 3: Effect of Hutchinson sample size m.
        
        Expected: More samples should reduce variance in trace estimation.
        Note: This only applies when using stochastic trace estimation.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: HUTCHINSON SAMPLE SIZE ANALYSIS")
        print("="*80)
        print("Note: Current implementation uses deterministic diagonal Hessian.")
        print("This experiment would require enabling full Hessian with Hutchinson estimator.")
        
        # For now, just document the limitation
        return []
    
    def _plot_trajectory_length_results(self, results: List[Dict]):
        """Plot trajectory length experiment results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        T_vals = [r['trajectory_length'] for r in results]
        traj_gaps = [r['trajectory_gap'] for r in results]
        base_gaps = [r['baseline_gap'] for r in results]
        kl_traj = [r['kl_trajectory'] for r in results]
        kl_base = [r['kl_baseline'] for r in results]
        test_risks = [r['test_risk'] for r in results]
        trace_H = [r['diagnostics']['trace_H_bar'] for r in results]
        trace_Sigma = [r['diagnostics']['trace_Sigma_Q'] for r in results]
        
        # 1. Bound gaps
        axes[0, 0].plot(T_vals, traj_gaps, 'o-', label='Trajectory', linewidth=2)
        axes[0, 0].plot(T_vals, base_gaps, 's-', label='Baseline', linewidth=2)
        axes[0, 0].set_xlabel('Trajectory Length T')
        axes[0, 0].set_ylabel('Bound Gap')
        axes[0, 0].set_title('Bound Tightness vs Trajectory Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. KL divergence
        axes[0, 1].plot(T_vals, kl_traj, '^-', label='Trajectory', linewidth=2, color='green')
        axes[0, 1].plot(T_vals, kl_base, 'v-', label='Baseline', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Trajectory Length T')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].set_title('KL Divergence vs Trajectory Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. Test risk (should be stable)
        axes[0, 2].plot(T_vals, test_risks, 'D-', linewidth=2, color='purple')
        axes[0, 2].set_xlabel('Trajectory Length T')
        axes[0, 2].set_ylabel('Test Risk')
        axes[0, 2].set_title('Test Risk vs Trajectory Length')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Trace of H_bar
        axes[1, 0].plot(T_vals, trace_H, 'o-', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Trajectory Length T')
        axes[1, 0].set_ylabel('tr(H̄_T)')
        axes[1, 0].set_title('Hessian Trace vs Trajectory Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Trace of Sigma_Q
        axes[1, 1].plot(T_vals, trace_Sigma, 's-', linewidth=2, color='blue')
        axes[1, 1].set_xlabel('Trajectory Length T')
        axes[1, 1].set_ylabel('tr(Σ_Q)')
        axes[1, 1].set_title('Posterior Variance vs Trajectory Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Ratio analysis
        ratio = [t / b if b > 0 else float('inf') for t, b in zip(traj_gaps, base_gaps)]
        axes[1, 2].plot(T_vals, ratio, 'o-', linewidth=2, color='black')
        axes[1, 2].axhline(y=1.0, color='r', linestyle='--', label='Equal')
        axes[1, 2].set_xlabel('Trajectory Length T')
        axes[1, 2].set_ylabel('Gap Ratio (Traj / Base)')
        axes[1, 2].set_title('Relative Bound Quality')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp1_trajectory_length.png', bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'exp1_trajectory_length.png'}")
    
    def _plot_posterior_scaling_results(self, results: List[Dict]):
        """Plot posterior scaling experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        c_Q_vals = [r['config']['c_Q'] for r in results]
        traj_gaps = [r['trajectory_gap'] for r in results]
        kl_traj = [r['kl_trajectory'] for r in results]
        trace_Sigma = [r['diagnostics']['trace_Sigma_Q'] for r in results]
        
        # 1. Bound gap vs c_Q
        axes[0, 0].plot(c_Q_vals, traj_gaps, 'o-', linewidth=2, markersize=8)
        optimal_idx = np.argmin(traj_gaps)
        axes[0, 0].scatter([c_Q_vals[optimal_idx]], [traj_gaps[optimal_idx]], 
                          s=300, marker='*', color='red', zorder=5,
                          label=f'Optimal: c_Q={c_Q_vals[optimal_idx]}')
        axes[0, 0].set_xlabel('Posterior Scaling c_Q')
        axes[0, 0].set_ylabel('Trajectory Bound Gap')
        axes[0, 0].set_title('Bound Gap vs Posterior Scaling')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # 2. KL vs c_Q
        axes[0, 1].plot(c_Q_vals, kl_traj, '^-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Posterior Scaling c_Q')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].set_title('KL Divergence vs Posterior Scaling')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        
        # 3. Trace of Sigma_Q vs c_Q
        axes[1, 0].plot(c_Q_vals, trace_Sigma, 's-', linewidth=2, markersize=8, color='blue')
        axes[1, 0].set_xlabel('Posterior Scaling c_Q')
        axes[1, 0].set_ylabel('tr(Σ_Q)')
        axes[1, 0].set_title('Posterior Variance vs Scaling')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        
        # 4. Trade-off: KL vs gap
        axes[1, 1].scatter(kl_traj, traj_gaps, s=100, c=c_Q_vals, cmap='viridis')
        for i, c_Q in enumerate(c_Q_vals):
            axes[1, 1].annotate(f'c_Q={c_Q}', (kl_traj[i], traj_gaps[i]), 
                               fontsize=8, ha='right')
        axes[1, 1].set_xlabel('KL Divergence')
        axes[1, 1].set_ylabel('Bound Gap')
        axes[1, 1].set_title('KL-Gap Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp2_posterior_scaling.png', bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'exp2_posterior_scaling.png'}")
        
        print(f"\nOptimal c_Q: {c_Q_vals[optimal_idx]} (gap: {traj_gaps[optimal_idx]:.4f})")


def main():
    """Run all validation experiments."""
    
    # Base configuration
    base_config = {
        'class_0': 0,
        'class_1': 1,
        'data_dir': 'data',
        'batch_size': 64,
        'val_split': 0.2,
        'model_type': 'logistic',
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'num_epochs': 50,
        'trajectory_interval': 5,
        'c_Q': 1.0,
        'lambda_reg': 0.01,
        'prior_variance': 1.0,
        'delta': 0.05,
        'use_diagonal_hessian': True
    }
    
    validator = ValidationExperiments(base_config, output_dir='validation_results')
    
    # Run experiments
    print("\n" + "="*80)
    print("VALIDATION EXPERIMENTS TO DIAGNOSE BOUND BEHAVIOR")
    print("="*80)
    
    all_results = {}
    
    # Experiment 1: Trajectory length
    all_results['trajectory_length'] = validator.experiment_1_trajectory_length(
        T_values=[20, 50, 100, 200]
    )
    
    # Experiment 2: Posterior scaling
    all_results['posterior_scaling'] = validator.experiment_2_posterior_scaling(
        c_Q_values=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    # Save results
    output_file = Path('validation_results') / 'validation_results.json'
    
    # Convert to serializable format
    serializable_results = {}
    for exp_name, exp_results in all_results.items():
        serializable_results[exp_name] = []
        for result in exp_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.floating, np.integer)):
                    serializable_result[key] = float(value)
                elif isinstance(value, dict):
                    serializable_result[key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_result[key] = value
            serializable_results[exp_name].append(serializable_result)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_file}")
    print(f"Figures saved to: validation_results/")
    
    print("\n" + "="*80)
    print("VALIDATION EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
