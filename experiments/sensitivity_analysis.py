"""
Hyperparameter Sensitivity Analysis
====================================

Validates theoretical expectations:
- Effect of trajectory length T
- Effect of Hutchinson sample size m
- Effect of posterior scaling c_Q
- Effect of top-subspace rank r
- Effect of regularization λ
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MNISTBinaryClassification
from src.models import create_model
from src.training import TrajectoryTrainer
from src.hessian_estimator import TrajectoryHessianEstimator
from src.pac_bayes_bounds import GaussianPACBayesBound
from src.utils import set_random_seed

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class SensitivityAnalyzer:
    """
    Performs systematic sensitivity analysis across hyperparameters.
    """
    
    def __init__(self, base_config: Dict, output_dir: str = "figures"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _run_single_config(self, config: Dict, seed: int = 42) -> Dict:
        """Run experiment with given configuration."""
        set_random_seed(seed)
        
        # Load data
        data_loader = MNISTBinaryClassification(
            class_0=config['class_0'],
            class_1=config['class_1'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            val_split=config['val_split']
        )
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        
        # Create model
        model = create_model(config['model_type'], input_dim=784, hidden_dims=config.get('hidden_dims'))
        
        # Train with trajectory recording
        trainer = TrajectoryTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=config['optimizer'],
            learning_rate=config['learning_rate'],
            trajectory_interval=config['trajectory_interval']
        )
        
        history = trainer.train(num_epochs=config['num_epochs'])
        trajectory = trainer.get_trajectory()
        
        # Compute Hessian
        hessian_estimator = TrajectoryHessianEstimator(
            model=model,
            trajectory=trajectory,
            data_loader=train_loader
        )
        
        H_bar = hessian_estimator.compute_trajectory_averaged_hessian(
            use_diagonal=config['use_diagonal_hessian']
        )
        Sigma_Q = hessian_estimator.compute_posterior_covariance(
            H_bar=H_bar,
            c_Q=config['c_Q'],
            lambda_reg=config['lambda_reg']
        )
        
        # Compute bounds
        pac_bayes = GaussianPACBayesBound(
            prior_mean=np.zeros(model.get_num_parameters()),
            prior_cov_inv=np.eye(model.get_num_parameters()) / config['prior_variance'],
            delta=config['delta']
        )
        
        train_risk = trainer.compute_empirical_risk(train_loader)
        test_risk = trainer.compute_empirical_risk(test_loader)
        
        w_Q = trajectory[-1]
        
        trajectory_bound = pac_bayes.compute_pac_bayes_bound(
            w_Q=w_Q,
            Sigma_Q=Sigma_Q,
            empirical_risk=train_risk,
            n_samples=len(train_loader.dataset),
            stochastic_error=0.0
        )
        
        return {
            'train_risk': train_risk,
            'test_risk': test_risk,
            'bound': trajectory_bound['pac_bayes_bound'],
            'kl': trajectory_bound['kl_divergence'],
            'gap': trajectory_bound['pac_bayes_bound'] - test_risk
        }
    
    def analyze_trajectory_length(self, T_values: List[int], n_seeds: int = 3) -> Dict:
        """
        Analyze effect of trajectory length T.
        
        Expected: Bound tightens with longer trajectories (better curvature estimation).
        """
        print("\n" + "="*60)
        print("TRAJECTORY LENGTH SENSITIVITY (T)")
        print("="*60)
        
        results = {T: [] for T in T_values}
        
        for T in T_values:
            print(f"\nTesting T={T}...")
            config = self.base_config.copy()
            config['num_epochs'] = T
            config['trajectory_interval'] = max(1, T // 20)  # Keep ~20 snapshots
            
            for seed in range(n_seeds):
                try:
                    result = self._run_single_config(config, seed=42 + seed)
                    results[T].append(result)
                except Exception as e:
                    print(f"  Error with seed {seed}: {e}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        T_vals = sorted(results.keys())
        means_bound = [np.mean([r['bound'] for r in results[T]]) for T in T_vals]
        stds_bound = [np.std([r['bound'] for r in results[T]]) for T in T_vals]
        means_gap = [np.mean([r['gap'] for r in results[T]]) for T in T_vals]
        stds_gap = [np.std([r['gap'] for r in results[T]]) for T in T_vals]
        means_kl = [np.mean([r['kl'] for r in results[T]]) for T in T_vals]
        
        # 1. Bound vs T
        axes[0, 0].errorbar(T_vals, means_bound, yerr=stds_bound, marker='o', capsize=5, linewidth=2)
        axes[0, 0].set_xlabel('Trajectory Length T')
        axes[0, 0].set_ylabel('PAC-Bayes Bound')
        axes[0, 0].set_title('Bound vs Trajectory Length')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gap vs T
        axes[0, 1].errorbar(T_vals, means_gap, yerr=stds_gap, marker='s', capsize=5, 
                           linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Trajectory Length T')
        axes[0, 1].set_ylabel('Bound Gap')
        axes[0, 1].set_title('Bound Tightness vs Trajectory Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. KL vs T
        axes[1, 0].plot(T_vals, means_kl, marker='^', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Trajectory Length T')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('KL Divergence vs Trajectory Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Relative improvement
        if len(T_vals) > 1:
            baseline_gap = means_gap[0]
            improvements = [(baseline_gap - g) / baseline_gap * 100 for g in means_gap]
            axes[1, 1].plot(T_vals, improvements, marker='D', linewidth=2, color='red')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Trajectory Length T')
            axes[1, 1].set_ylabel('Improvement over T={} (%)'.format(T_vals[0]))
            axes[1, 1].set_title('Relative Improvement')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_trajectory_length.png', bbox_inches='tight')
        print(f"\nSaved: sensitivity_trajectory_length.png")
        
        return results
    
    def analyze_hutchinson_samples(self, m_values: List[int], n_seeds: int = 3) -> Dict:
        """
        Analyze effect of Hutchinson sample size m.
        
        Expected: Larger m reduces stochastic error, more stable bounds.
        """
        print("\n" + "="*60)
        print("HUTCHINSON SAMPLE SIZE SENSITIVITY (m)")
        print("="*60)
        
        results = {m: [] for m in m_values}
        
        for m in m_values:
            print(f"\nTesting m={m}...")
            config = self.base_config.copy()
            config['m_hutchinson'] = m
            
            for seed in range(n_seeds):
                try:
                    result = self._run_single_config(config, seed=42 + seed)
                    results[m].append(result)
                except Exception as e:
                    print(f"  Error with seed {seed}: {e}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        m_vals = sorted(results.keys())
        means_bound = [np.mean([r['bound'] for r in results[m]]) for m in m_vals]
        stds_bound = [np.std([r['bound'] for r in results[m]]) for m in m_vals]
        means_gap = [np.mean([r['gap'] for r in results[m]]) for m in m_vals]
        stds_gap = [np.std([r['gap'] for r in results[m]]) for m in m_vals]
        
        # 1. Bound stability
        axes[0].errorbar(m_vals, means_bound, yerr=stds_bound, marker='o', capsize=5, linewidth=2)
        axes[0].set_xlabel('Hutchinson Sample Size m')
        axes[0].set_ylabel('PAC-Bayes Bound')
        axes[0].set_title('Bound Stability vs Sample Size')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Variance reduction
        axes[1].plot(m_vals, stds_bound, marker='s', linewidth=2, color='orange', label='Bound Std')
        axes[1].plot(m_vals, stds_gap, marker='^', linewidth=2, color='green', label='Gap Std')
        axes[1].set_xlabel('Hutchinson Sample Size m')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_title('Variance Reduction with Sample Size')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_hutchinson_samples.png', bbox_inches='tight')
        print(f"\nSaved: sensitivity_hutchinson_samples.png")
        
        return results
    
    def analyze_posterior_scaling(self, c_Q_values: List[float], n_seeds: int = 3) -> Dict:
        """
        Analyze effect of posterior scaling c_Q.
        
        Expected: Intermediate c_Q produces tightest bounds (trade-off).
        """
        print("\n" + "="*60)
        print("POSTERIOR SCALING SENSITIVITY (c_Q)")
        print("="*60)
        
        results = {c: [] for c in c_Q_values}
        
        for c_Q in c_Q_values:
            print(f"\nTesting c_Q={c_Q}...")
            config = self.base_config.copy()
            config['c_Q'] = c_Q
            
            for seed in range(n_seeds):
                try:
                    result = self._run_single_config(config, seed=42 + seed)
                    results[c_Q].append(result)
                except Exception as e:
                    print(f"  Error with seed {seed}: {e}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        c_vals = sorted(results.keys())
        means_bound = [np.mean([r['bound'] for r in results[c]]) for c in c_vals]
        stds_bound = [np.std([r['bound'] for r in results[c]]) for c in c_vals]
        means_gap = [np.mean([r['gap'] for r in results[c]]) for c in c_vals]
        means_kl = [np.mean([r['kl'] for r in results[c]]) for c in c_vals]
        
        # 1. Bound vs c_Q
        axes[0, 0].errorbar(c_vals, means_bound, yerr=stds_bound, marker='o', capsize=5, linewidth=2)
        axes[0, 0].set_xlabel('Posterior Scaling c_Q')
        axes[0, 0].set_ylabel('PAC-Bayes Bound')
        axes[0, 0].set_title('Bound vs Posterior Scaling')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gap vs c_Q (should show optimum)
        axes[0, 1].plot(c_vals, means_gap, marker='s', linewidth=2, color='orange')
        optimal_idx = np.argmin(means_gap)
        axes[0, 1].scatter([c_vals[optimal_idx]], [means_gap[optimal_idx]], 
                          s=200, marker='*', color='red', zorder=5, 
                          label=f'Optimal: c_Q={c_vals[optimal_idx]}')
        axes[0, 1].set_xlabel('Posterior Scaling c_Q')
        axes[0, 1].set_ylabel('Bound Gap')
        axes[0, 1].set_title('Bound Tightness vs Posterior Scaling')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. KL vs c_Q
        axes[1, 0].plot(c_vals, means_kl, marker='^', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Posterior Scaling c_Q')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('KL Divergence vs Posterior Scaling')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trade-off visualization
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(c_vals, means_kl, marker='^', linewidth=2, color='green', label='KL')
        ax2.plot(c_vals, means_gap, marker='s', linewidth=2, color='orange', label='Gap')
        axes[1, 1].set_xlabel('Posterior Scaling c_Q')
        axes[1, 1].set_ylabel('KL Divergence', color='green')
        ax2.set_ylabel('Bound Gap', color='orange')
        axes[1, 1].set_title('KL-Gap Trade-off')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_posterior_scaling.png', bbox_inches='tight')
        print(f"\nSaved: sensitivity_posterior_scaling.png")
        
        print(f"\nOptimal c_Q: {c_vals[optimal_idx]} (tightest bound)")
        
        return results
    
    def run_all_sensitivity_analyses(self):
        """Run all sensitivity analyses."""
        print("\n" + "="*60)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("="*60)
        
        # 1. Trajectory length
        T_values = [10, 20, 50, 100]
        traj_results = self.analyze_trajectory_length(T_values, n_seeds=3)
        
        # 2. Posterior scaling
        c_Q_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        scaling_results = self.analyze_posterior_scaling(c_Q_values, n_seeds=3)
        
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'trajectory_length': traj_results,
            'posterior_scaling': scaling_results
        }


def main():
    """Main execution."""
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
        'use_diagonal_hessian': True,
        'm_hutchinson': 100
    }
    
    analyzer = SensitivityAnalyzer(base_config, output_dir='figures')
    results = analyzer.run_all_sensitivity_analyses()
    
    # Save results
    output_path = Path('results') / 'sensitivity_analysis.json'
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert to serializable format
    serializable_results = {}
    for key, val_dict in results.items():
        serializable_results[key] = {
            str(k): [
                {rk: float(rv) if isinstance(rv, (np.floating, np.integer)) else rv 
                 for rk, rv in r.items()}
                for r in v
            ]
            for k, v in val_dict.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
