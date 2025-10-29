"""
Unified Parallel Experiment Runner
===================================

Combines all experiments into one parallelized run:
1. Original experiments (baseline setup)
2. Trajectory length sensitivity (T)
3. Posterior scaling sensitivity (c_Q)
4. Hutchinson samples sensitivity (m)
5. Regularization sensitivity (λ)

Runs configurations in parallel and saves results incrementally.
Generates all figures and analysis files at the end.

Usage:
    # Quick test (small subset)
    python experiments/unified_experiments.py --mode test --workers 4
    
    # Full validation run
    python experiments/unified_experiments.py --mode validation --workers 8
    
    # Extensive overnight sweep
    python experiments/unified_experiments.py --mode extensive --workers 16
    
    # Custom configuration
    python experiments/unified_experiments.py --mode custom \
        --T 20,50,100,200 --c_Q 0.1,0.5,1.0,2.0 --workers 8
"""

import numpy as np
import json
import sys
from pathlib import Path
import torch
import argparse
from datetime import datetime
import tempfile
import uuid
import os
from multiprocessing import Pool, cpu_count
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MNISTBinaryClassification
from src.models import create_model
from src.training import TrajectoryTrainer
from src.hessian_estimator import TrajectoryHessianEstimator
from src.pac_bayes_bounds import GaussianPACBayesBound
from src.utils import set_random_seed


# ============================================================================
# Configuration Presets
# ============================================================================

PRESETS = {
    'test': {
        'description': 'Quick test (2-3 minutes, 6 configs)',
        'T': [20, 50],
        'c_Q': [0.5, 1.0],
        'lambda': [0.01],
        'm': [100],
        'model_types': ['logistic'],
        'seeds': [42]
    },
    'validation': {
        'description': 'Full validation from paper (30-60 minutes, ~40 configs)',
        'T': [20, 50, 100, 200],
        'c_Q': [0.1, 0.5, 1.0, 2.0, 5.0],
        'lambda': [0.001, 0.01, 0.1],
        'm': [50, 100, 200, 500],
        'model_types': ['logistic'],
        'seeds': [42]
    },
    'extensive': {
        'description': 'Extensive overnight sweep (8-12 hours, ~500 configs)',
        'T': [10, 20, 50, 100, 200, 300],
        'c_Q': [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'lambda': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'm': [50, 100, 200, 500, 1000],
        'model_types': ['logistic'],
        'seeds': [42, 123, 456]  # Multiple seeds for robustness
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def _serialize_value(v):
    """Convert numpy/torch types to native Python types."""
    try:
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if hasattr(v, 'cpu') and hasattr(v, 'numpy'):
            return _serialize_value(v.cpu().numpy())
        if isinstance(v, (list, tuple)):
            return [_serialize_value(x) for x in v]
        return v
    except Exception:
        return str(v)


def _serialize_result(d: dict) -> dict:
    return {k: _serialize_value(v) for k, v in d.items()}


def _atomic_append_json(item: dict, out_path: Path):
    """Append a single item to a JSON list atomically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use lock file for process-safe writes
    lock_file = out_path.with_suffix('.lock')
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            # Try to create lock file exclusively
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            
            try:
                # Load existing data
                if out_path.exists():
                    with open(out_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = [data]
                        except Exception:
                            data = []
                else:
                    data = []
                
                # Append new item
                data.append(item)
                
                # Write atomically via temp file
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=str(out_path.parent), 
                    suffix='.json'
                )
                try:
                    with open(temp_path, 'w') as tf:
                        json.dump(data, tf, indent=2)
                    Path(temp_path).replace(out_path)
                finally:
                    try:
                        os.close(temp_fd)
                    except:
                        pass
                
                break  # Success
                
            finally:
                # Remove lock
                try:
                    lock_file.unlink()
                except:
                    pass
                    
        except FileExistsError:
            # Lock exists, wait and retry
            import time
            time.sleep(0.1 * (attempt + 1))
            continue
    
    else:
        raise RuntimeError(f"Failed to acquire lock after {max_attempts} attempts")


# ============================================================================
# Core Experiment Runner
# ============================================================================

def run_single_experiment(config_tuple):
    """
    Run a single experiment configuration.
    
    Parameters:
    -----------
    config_tuple : tuple
        (model_type, T, c_Q, lambda_reg, m_hess, seed, out_file, worker_id, config_id, total_configs)
    
    Returns:
    --------
    result : dict
        Complete experiment results with all metrics
    """
    (model_type, T, c_Q, lambda_reg, m_hess, seed, 
     out_file, worker_id, config_id, total_configs) = config_tuple
    
    start_time = datetime.now()
    
    try:
        set_random_seed(seed)
        
        # Load data (cached after first load)
        data_loader = MNISTBinaryClassification(
            class_0=0, class_1=1,
            data_dir='data',
            val_split=0.15
        )
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=64)
        
        # Create model
        model = create_model(model_type, input_dim=784)
        d = len(model.get_parameters_flat())
        
        # Train with trajectory recording
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        record_every = max(1, T // 20)  # ~20 snapshots
        trainer = TrajectoryTrainer(
            model=model, optimizer=optimizer, criterion=criterion,
            device='cpu', record_every=record_every, verbose=False
        )
        
        trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=T)
        trajectory = [p.cpu().detach().clone() for p in trainer.trajectory]
        
        # Compute Hessian and posterior
        hess_est = TrajectoryHessianEstimator(
            model=model, trajectory=trajectory, data_loader=train_loader,
            criterion=criterion, device='cpu',
            lambda_reg=lambda_reg, c_Q=c_Q,
            use_diagonal=True, num_hess_samples=m_hess, verbose=False
        )
        
        H_bar = hess_est.H_bar.cpu().numpy()
        Sigma_Q = hess_est.Sigma_Q.cpu().numpy()
        
        # Compute risks
        w_Q = trajectory[-1]
        train_risk = trainer.compute_empirical_risk(train_loader)
        val_risk = trainer.compute_empirical_risk(val_loader)
        test_risk = trainer.compute_empirical_risk(test_loader)
        
        # Setup prior and posterior
        mu_P = torch.zeros(d)
        Sigma_P = torch.ones(d)  # N(0, I)
        
        # Trajectory-aware bound
        pac_bayes_traj = GaussianPACBayesBound(
            mu_P=mu_P, Sigma_P=Sigma_P, mu_Q=w_Q,
            Sigma_Q=torch.from_numpy(Sigma_Q).float(),
            r=10, device='cpu', use_diagonal=True
        )
        traj_kl = pac_bayes_traj.compute_gaussian_kl()
        
        n = len(train_loader.dataset)
        delta = 0.05
        traj_bound = train_risk + np.sqrt((traj_kl + np.log(1/delta)) / (2*n))
        
        # Baseline bound (isotropic posterior)
        Sigma_baseline = torch.ones(d)
        pac_bayes_base = GaussianPACBayesBound(
            mu_P=mu_P, Sigma_P=Sigma_P, mu_Q=w_Q,
            Sigma_Q=Sigma_baseline,
            r=10, device='cpu', use_diagonal=True
        )
        base_kl = pac_bayes_base.compute_gaussian_kl()
        base_bound = train_risk + np.sqrt((base_kl + np.log(1/delta)) / (2*n))
        
        # Compute gaps for convenience
        traj_gap = traj_bound - test_risk
        base_gap = base_bound - test_risk
        
        # KL decomposition (if available)
        try:
            kl_decomp = pac_bayes_traj.compute_kl_decomposition()
        except:
            kl_decomp = {}
        
        # Compute training history statistics
        train_losses = trainer.train_losses
        val_losses = trainer.val_losses
        train_accs = trainer.train_accs
        val_accs = trainer.val_accs
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile complete result
        result = {
            # Metadata
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'worker_id': int(worker_id),
            'config_id': int(config_id),
            'duration_seconds': float(duration),
            'status': 'success',
            
            # Configuration
            'model_type': model_type,
            'seed': int(seed),
            'T': int(T),
            'c_Q': float(c_Q),
            'lambda': float(lambda_reg),
            'm': int(m_hess),
            'num_parameters': int(d),
            'trajectory_length': len(trajectory),
            
            # Risks
            'train_risk': float(train_risk),
            'val_risk': float(val_risk),
            'test_risk': float(test_risk),
            
            # Bounds
            'traj_bound': float(traj_bound),
            'traj_kl': float(traj_kl),
            'traj_gap': float(traj_bound - test_risk),
            'base_bound': float(base_bound),
            'base_kl': float(base_kl),
            'base_gap': float(base_bound - test_risk),
            
            # Bound improvement
            'bound_improvement': float(base_bound - traj_bound),
            'gap_improvement': float(base_gap - traj_gap),
            'improvement_percent': float((base_gap - traj_gap) / base_gap * 100) if base_gap > 0 else 0.0,
            
            # Hessian/Covariance statistics
            'trace_H_bar': float(np.sum(H_bar)),
            'mean_H_bar': float(np.mean(H_bar)),
            'max_H_bar': float(np.max(H_bar)),
            'min_H_bar': float(np.min(H_bar)),
            'trace_Sigma_Q': float(np.sum(Sigma_Q)),
            'mean_Sigma_Q': float(np.mean(Sigma_Q)),
            'max_Sigma_Q': float(np.max(Sigma_Q)),
            'min_Sigma_Q': float(np.min(Sigma_Q)),
            
            # Training history summary
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'final_train_acc': float(train_accs[-1]),
            'final_val_acc': float(val_accs[-1]),
            'best_val_acc': float(max(val_accs)),
            'train_loss_reduction': float(train_losses[0] - train_losses[-1]),
            
            # KL decomposition (if available)
            'kl_decomposition': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                for k, v in kl_decomp.items()},
        }
        
        # Save immediately
        _atomic_append_json(_serialize_result(result), Path(out_file))
        
        # Progress output
        print(f"[{config_id}/{total_configs}] [Worker {worker_id}] ✓ "
              f"T={T:3d} c_Q={c_Q:4.1f} λ={lambda_reg:.4f} m={m_hess:4d} | "
              f"Gap: {result['traj_gap']:.4f} | Improve: {result['improvement_percent']:+6.1f}% | "
              f"{duration:.1f}s | {datetime.now().strftime('%H:%M:%S')}")
        
        return result
        
    except Exception as e:
        import traceback
        error_result = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'worker_id': int(worker_id),
            'config_id': int(config_id),
            'model_type': model_type,
            'T': int(T),
            'c_Q': float(c_Q),
            'lambda': float(lambda_reg),
            'm': int(m_hess),
            'seed': int(seed),
            'status': 'error',
            'error_message': str(e),
            'error_traceback': traceback.format_exc()
        }
        
        _atomic_append_json(_serialize_result(error_result), Path(out_file))
        
        print(f"[{config_id}/{total_configs}] [Worker {worker_id}] ✗ ERROR "
              f"T={T} c_Q={c_Q}: {e}")
        
        return error_result


# ============================================================================
# Analysis and Visualization
# ============================================================================

def generate_all_analyses(results_file: Path, output_dir: Path):
    """Generate all figures and analysis files from results."""
    print("\n" + "="*70)
    print("GENERATING ANALYSES AND FIGURES")
    print("="*70)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Filter successful results
    results = [r for r in results if r.get('status') == 'success']
    
    if not results:
        print("No successful results to analyze!")
        return
    
    df = pd.DataFrame(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save summary statistics
    print("\n1. Generating summary statistics...")
    summary_stats = df.describe().T
    summary_stats.to_csv(output_dir / 'summary_statistics.csv')
    print(f"   Saved: {output_dir / 'summary_statistics.csv'}")
    
    # 2. Save full results table
    print("\n2. Saving full results table...")
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"   Saved: {output_dir / 'all_results.csv'}")
    
    # 3. Trajectory length analysis
    if len(df['T'].unique()) > 1:
        print("\n3. Trajectory length analysis...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        T_grouped = df.groupby('T').agg({
            'traj_gap': ['mean', 'std'],
            'base_gap': ['mean', 'std'],
            'improvement_percent': ['mean', 'std']
        }).reset_index()
        
        T_vals = T_grouped['T'].values
        
        axes[0].errorbar(T_vals, T_grouped[('traj_gap', 'mean')], 
                        yerr=T_grouped[('traj_gap', 'std')], 
                        marker='o', label='Trajectory', linewidth=2, capsize=5)
        axes[0].errorbar(T_vals, T_grouped[('base_gap', 'mean')], 
                        yerr=T_grouped[('base_gap', 'std')], 
                        marker='s', label='Baseline', linewidth=2, capsize=5)
        axes[0].set_xlabel('Trajectory Length T')
        axes[0].set_ylabel('Bound Gap')
        axes[0].set_title('Bound Tightness vs Trajectory Length')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].errorbar(T_vals, T_grouped[('improvement_percent', 'mean')],
                        yerr=T_grouped[('improvement_percent', 'std')],
                        marker='D', color='green', linewidth=2, capsize=5)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Trajectory Length T')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Bound Improvement vs T')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].scatter(df['T'], df['traj_gap'], alpha=0.3, s=20)
        axes[2].set_xlabel('Trajectory Length T')
        axes[2].set_ylabel('Trajectory Bound Gap')
        axes[2].set_title('All Runs: Gap vs T')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_trajectory_length.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: {output_dir / 'analysis_trajectory_length.png'}")
        plt.close()
    
    # 4. Posterior scaling analysis
    if len(df['c_Q'].unique()) > 1:
        print("\n4. Posterior scaling analysis...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        cQ_grouped = df.groupby('c_Q').agg({
            'traj_gap': ['mean', 'std'],
            'traj_kl': ['mean', 'std'],
            'trace_Sigma_Q': ['mean', 'std']
        }).reset_index()
        
        cQ_vals = cQ_grouped['c_Q'].values
        
        axes[0].errorbar(cQ_vals, cQ_grouped[('traj_gap', 'mean')],
                        yerr=cQ_grouped[('traj_gap', 'std')],
                        marker='o', linewidth=2, capsize=5)
        axes[0].set_xlabel('Posterior Scaling c_Q')
        axes[0].set_ylabel('Bound Gap')
        axes[0].set_title('Bound Tightness vs c_Q')
        axes[0].grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_idx = cQ_grouped[('traj_gap', 'mean')].idxmin()
        axes[0].scatter([cQ_vals[optimal_idx]], [cQ_grouped[('traj_gap', 'mean')].iloc[optimal_idx]],
                       s=200, marker='*', color='red', zorder=5, label=f'Optimal: {cQ_vals[optimal_idx]:.2f}')
        axes[0].legend()
        
        axes[1].errorbar(cQ_vals, cQ_grouped[('traj_kl', 'mean')],
                        yerr=cQ_grouped[('traj_kl', 'std')],
                        marker='s', color='orange', linewidth=2, capsize=5)
        axes[1].set_xlabel('Posterior Scaling c_Q')
        axes[1].set_ylabel('KL Divergence')
        axes[1].set_title('KL vs c_Q')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].errorbar(cQ_vals, cQ_grouped[('trace_Sigma_Q', 'mean')],
                        yerr=cQ_grouped[('trace_Sigma_Q', 'std')],
                        marker='^', color='green', linewidth=2, capsize=5)
        axes[2].set_xlabel('Posterior Scaling c_Q')
        axes[2].set_ylabel('Trace(Σ_Q)')
        axes[2].set_title('Posterior Variance vs c_Q')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_posterior_scaling.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: {output_dir / 'analysis_posterior_scaling.png'}")
        plt.close()
    
    # 5. Hutchinson samples analysis
    if len(df['m'].unique()) > 1:
        print("\n5. Hutchinson samples analysis...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        m_grouped = df.groupby('m').agg({
            'traj_gap': ['mean', 'std'],
            'trace_H_bar': ['mean', 'std']
        }).reset_index()
        
        m_vals = m_grouped['m'].values
        
        axes[0].errorbar(m_vals, m_grouped[('traj_gap', 'mean')],
                        yerr=m_grouped[('traj_gap', 'std')],
                        marker='o', linewidth=2, capsize=5)
        axes[0].set_xlabel('Hutchinson Sample Size m')
        axes[0].set_ylabel('Bound Gap')
        axes[0].set_title('Bound Stability vs Sample Size')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].errorbar(m_vals, m_grouped[('trace_H_bar', 'mean')],
                        yerr=m_grouped[('trace_H_bar', 'std')],
                        marker='s', color='orange', linewidth=2, capsize=5)
        axes[1].set_xlabel('Hutchinson Sample Size m')
        axes[1].set_ylabel('Trace(H̄_T)')
        axes[1].set_title('Curvature Estimate vs Sample Size')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_hutchinson_samples.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: {output_dir / 'analysis_hutchinson_samples.png'}")
        plt.close()
    
    # 6. Overall comparison
    print("\n6. Overall bound comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(df['test_risk'], df['traj_bound'], alpha=0.5, s=30, label='Trajectory')
    axes[0].scatter(df['test_risk'], df['base_bound'], alpha=0.5, s=30, label='Baseline')
    axes[0].plot([0, df['test_risk'].max()], [0, df['test_risk'].max()], 
                'k--', alpha=0.3, label='Perfect bound')
    axes[0].set_xlabel('Test Risk')
    axes[0].set_ylabel('Bound Value')
    axes[0].set_title('Bound vs Test Risk')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df['improvement_percent'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    axes[1].axvline(x=df['improvement_percent'].mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["improvement_percent"].mean():.1f}%')
    axes[1].set_xlabel('Improvement (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Bound Improvements')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'analysis_overall_comparison.png'}")
    plt.close()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated {len([f for f in output_dir.glob('*.png')])} figures")
    print(f"Generated {len([f for f in output_dir.glob('*.csv')])} CSV files")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified parallel experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Preset Modes:
{chr(10).join(f"  {name:12s} - {info['description']}" for name, info in PRESETS.items())}

Examples:
  # Quick test
  python experiments/unified_experiments.py --mode test --workers 4
  
  # Full validation
  python experiments/unified_experiments.py --mode validation --workers 8 --out results/validation.json
  
  # Overnight extensive run
  nohup python experiments/unified_experiments.py --mode extensive --workers 16 \\
      --out results/extensive_$(date +%%Y%%m%%d).json > logs/run.log 2>&1 &
  
  # Custom configuration
  python experiments/unified_experiments.py --mode custom \\
      --T 20,50,100 --c_Q 0.5,1.0,2.0 --lambda 0.01,0.1 --m 100,200 --workers 8
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=list(PRESETS.keys()) + ['custom'],
                       help='Experiment mode (test/validation/extensive/custom)')
    parser.add_argument('--out', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    
    # Custom configuration options
    parser.add_argument('--T', type=str, default=None,
                       help='Comma-separated trajectory lengths (for custom mode)')
    parser.add_argument('--c_Q', type=str, default=None,
                       help='Comma-separated c_Q values (for custom mode)')
    parser.add_argument('--lambda', dest='lambda_reg', type=str, default=None,
                       help='Comma-separated lambda values (for custom mode)')
    parser.add_argument('--m', type=str, default=None,
                       help='Comma-separated Hutchinson sample sizes (for custom mode)')
    parser.add_argument('--model', type=str, default=None,
                       help='Comma-separated model types (for custom mode)')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated random seeds (for custom mode)')
    
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip automatic analysis/figure generation')
    
    args = parser.parse_args()
    
    # Load or build configuration
    if args.mode == 'custom':
        if not all([args.T, args.c_Q, args.lambda_reg, args.m]):
            parser.error("Custom mode requires --T, --c_Q, --lambda, and --m arguments")
        
        config = {
            'T': [int(x) for x in args.T.split(',')],
            'c_Q': [float(x) for x in args.c_Q.split(',')],
            'lambda': [float(x) for x in args.lambda_reg.split(',')],
            'm': [int(x) for x in args.m.split(',')],
            'model_types': args.model.split(',') if args.model else ['logistic'],
            'seeds': [int(x) for x in args.seeds.split(',')] if args.seeds else [42]
        }
    else:
        config = PRESETS[args.mode]
    
    # Setup output paths
    if args.out is None:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_file = Path('results') / f'unified_{args.mode}_{timestamp}.json'
    else:
        out_file = Path(args.out)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir = out_file.parent / 'analysis'
    
    # Generate all configurations
    configs = list(product(
        config['model_types'],
        config['T'],
        config['c_Q'],
        config['lambda'],
        config['m'],
        config['seeds']
    ))
    
    n_configs = len(configs)
    n_workers = args.workers if args.workers else cpu_count()
    n_workers = min(n_workers, n_configs)
    
    # Prepare config tuples
    config_tuples = [
        (model, T, c_Q, lam, m, seed, str(out_file), i % n_workers, i+1, n_configs)
        for i, (model, T, c_Q, lam, m, seed) in enumerate(configs)
    ]
    
    # Print header
    start_time = datetime.now()
    print("\n" + "="*70)
    print("UNIFIED PARALLEL EXPERIMENT RUNNER")
    print("="*70)
    print(f"Mode: {args.mode}")
    if args.mode in PRESETS:
        print(f"Description: {PRESETS[args.mode]['description']}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {out_file}")
    print(f"Workers: {n_workers} (of {cpu_count()} CPUs)")
    print(f"Total configurations: {n_configs}")
    print(f"  Model types: {config['model_types']}")
    print(f"  T values: {config['T']}")
    print(f"  c_Q values: {config['c_Q']}")
    print(f"  λ values: {config['lambda']}")
    print(f"  m values: {config['m']}")
    print(f"  Seeds: {config['seeds']}")
    print("="*70)
    
    # Run experiments in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_experiment, config_tuples)
    
    # Summary statistics
    n_success = sum(1 for r in results if r.get('status') == 'success')
    n_error = sum(1 for r in results if r.get('status') == 'error')
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Success:  {n_success}/{n_configs} ({100*n_success/n_configs:.1f}%)")
    if n_error > 0:
        print(f"Errors:   {n_error}/{n_configs} ({100*n_error/n_configs:.1f}%)")
    print(f"\nResults saved to: {out_file}")
    
    # Generate analyses
    if not args.skip_analysis and n_success > 0:
        try:
            generate_all_analyses(out_file, output_dir)
            print(f"\nAnalysis outputs in: {output_dir}")
        except Exception as e:
            print(f"\nWarning: Analysis generation failed: {e}")
            print("You can run analysis manually later with:")
            print(f"  python experiments/run_comprehensive_analysis.py {out_file}")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
