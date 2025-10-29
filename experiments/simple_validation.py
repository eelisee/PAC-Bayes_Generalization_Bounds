"""
Simple Validation Experiments
==============================

Run the key validation experiments from the paper analysis:
1. Trajectory length effect (T ∈ {20, 50, 100, 200})
2. Posterior scaling effect (c_Q ∈ {0.1, 0.5, 1.0, 2.0, 5.0})
3. Hutchinson samples effect (m ∈ {50, 100, 200, 500})
"""

import numpy as np
import json
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import tempfile
import uuid
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MNISTBinaryClassification
from src.models import create_model
from src.training import TrajectoryTrainer
from src.hessian_estimator import TrajectoryHessianEstimator
from src.pac_bayes_bounds import GaussianPACBayesBound
from src.utils import set_random_seed


def _serialize_value(v):
    """Convert numpy/torch types to native Python types for JSON serialization."""
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
    """Append a single item to a JSON list stored at out_path atomically.

    If file doesn't exist, creates a new list with the item.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(out_path.parent))
    try:
        # Load existing
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

        data.append(item)

        with open(temp_path, 'w') as tf:
            json.dump(data, tf, indent=2)

        # atomic rename
        Path(temp_path).replace(out_path)
    finally:
        try:
            os.close(temp_fd)
        except Exception:
            pass


def run_single_experiment(T, c_Q, lambda_reg, m_hess, seed=42):
    """Run a single experiment with given hyperparameters."""
    set_random_seed(seed)
    
    # Load data
    data_loader = MNISTBinaryClassification(
        class_0=0, class_1=1,
        data_dir='data',
        val_split=0.15
    )
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=64)
    
    # Create model
    model = create_model('logistic', input_dim=784)
    d = len(model.get_parameters_flat())
    
    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    record_every = max(1, T // 20)  # ~20 snapshots
    trainer = TrajectoryTrainer(
        model=model, optimizer=optimizer, criterion=criterion,
        device='cpu', record_every=record_every
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
    
    # Compute bounds
    w_Q = trajectory[-1]
    train_risk = trainer.compute_empirical_risk(train_loader)
    test_risk = trainer.compute_empirical_risk(test_loader)
    
    # Trajectory bound
    mu_P = torch.zeros(d)
    Sigma_P = torch.ones(d)  # N(0, I) - diagonal
    
    pac_bayes_traj = GaussianPACBayesBound(
        mu_P=mu_P,
        Sigma_P=Sigma_P,
        mu_Q=w_Q,
        Sigma_Q=torch.from_numpy(Sigma_Q).float(),
        r=10,
        device='cpu',
        use_diagonal=True
    )
    
    traj_kl = pac_bayes_traj.compute_gaussian_kl()
    n = len(train_loader.dataset)
    delta = 0.05
    traj_bound_val = train_risk + np.sqrt((traj_kl + np.log(1/delta)) / (2*n))
    
    # Baseline bound (isotropic)
    Sigma_baseline = torch.ones(d)
    pac_bayes_base = GaussianPACBayesBound(
        mu_P=mu_P,
        Sigma_P=Sigma_P,
        mu_Q=w_Q,
        Sigma_Q=Sigma_baseline,
        r=10,
        device='cpu',
        use_diagonal=True
    )
    
    base_kl = pac_bayes_base.compute_gaussian_kl()
    base_bound_val = train_risk + np.sqrt((base_kl + np.log(1/delta)) / (2*n))
    
    # Attach metadata
    result = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'seed': int(seed),
        'T': int(T),
        'c_Q': float(c_Q),
        'lambda': float(lambda_reg),
        'm': int(m_hess),
        'train_risk': float(train_risk),
        'test_risk': float(test_risk),
        'traj_bound': float(traj_bound_val),
        'traj_kl': float(traj_kl),
        'traj_gap': float(traj_bound_val - test_risk),
        'base_bound': float(base_bound_val),
        'base_kl': float(base_kl),
        'base_gap': float(base_bound_val - test_risk),
        'trace_H_bar': float(np.sum(H_bar)),
        'trace_Sigma_Q': float(np.sum(Sigma_Q)),
        'final_train_acc': float(trainer.train_accs[-1]),
        'final_val_acc': float(trainer.val_accs[-1])
    }

    return result


def experiment_1_trajectory_length(out_file: Path):
    """Experiment 1: Effect of trajectory length."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: TRAJECTORY LENGTH")
    print("="*70)
    
    T_values = [20, 50, 100, 200]
    results = []
    
    for T in T_values:
        print(f"\nTesting T={T}...")
        result = run_single_experiment(T=T, c_Q=1.0, lambda_reg=0.01, m_hess=100)
        results.append(result)
        # Save result incrementally
        _atomic_append_json(_serialize_result(result), out_file)

        print(f"  Train risk: {result['train_risk']:.4f}")
        print(f"  Test risk: {result['test_risk']:.4f}")
        print(f"  Traj bound: {result['traj_bound']:.4f} (gap: {result['traj_gap']:.4f})")
        print(f"  Base bound: {result['base_bound']:.4f} (gap: {result['base_gap']:.4f})")
        print(f"  Improvement: {((result['base_gap'] - result['traj_gap']) / result['base_gap'] * 100):.1f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    T_vals = [r['T'] for r in results]
    traj_gaps = [r['traj_gap'] for r in results]
    base_gaps = [r['base_gap'] for r in results]
    
    axes[0].plot(T_vals, traj_gaps, 'o-', label='Trajectory bound gap', linewidth=2)
    axes[0].plot(T_vals, base_gaps, 's-', label='Baseline bound gap', linewidth=2)
    axes[0].set_xlabel('Trajectory Length T')
    axes[0].set_ylabel('Bound Gap (Bound - Test Risk)')
    axes[0].set_title('Bound Tightness vs Trajectory Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    improvements = [((r['base_gap'] - r['traj_gap']) / r['base_gap'] * 100) for r in results]
    axes[1].plot(T_vals, improvements, 'D-', color='green', linewidth=2)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Trajectory Length T')
    axes[1].set_ylabel('Improvement over Baseline (%)')
    axes[1].set_title('Relative Improvement')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/validation_trajectory_length.png', dpi=300, bbox_inches='tight')
    print("\nSaved: figures/validation_trajectory_length.png")
    
    return results


def experiment_2_posterior_scaling(out_file: Path):
    """Experiment 2: Effect of posterior scaling c_Q."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: POSTERIOR SCALING")
    print("="*70)
    
    c_Q_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    for c_Q in c_Q_values:
        print(f"\nTesting c_Q={c_Q}...")
        result = run_single_experiment(T=50, c_Q=c_Q, lambda_reg=0.01, m_hess=100)
        results.append(result)
        _atomic_append_json(_serialize_result(result), out_file)

        print(f"  Test risk: {result['test_risk']:.4f}")
        print(f"  Traj bound: {result['traj_bound']:.4f} (gap: {result['traj_gap']:.4f})")
        print(f"  Traj KL: {result['traj_kl']:.2f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    c_vals = [r['c_Q'] for r in results]
    gaps = [r['traj_gap'] for r in results]
    kls = [r['traj_kl'] for r in results]
    traces = [r['trace_Sigma_Q'] for r in results]
    
    axes[0].plot(c_vals, gaps, 'o-', linewidth=2)
    axes[0].set_xlabel('Posterior Scaling c_Q')
    axes[0].set_ylabel('Bound Gap')
    axes[0].set_title('Bound Tightness vs c_Q')
    axes[0].grid(True, alpha=0.3)
    optimal_idx = np.argmin(gaps)
    axes[0].scatter([c_vals[optimal_idx]], [gaps[optimal_idx]], 
                   s=200, marker='*', color='red', zorder=5)
    
    axes[1].plot(c_vals, kls, 's-', color='orange', linewidth=2)
    axes[1].set_xlabel('Posterior Scaling c_Q')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL vs c_Q')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(c_vals, traces, '^-', color='green', linewidth=2)
    axes[2].set_xlabel('Posterior Scaling c_Q')
    axes[2].set_ylabel('Trace(Σ_Q)')
    axes[2].set_title('Posterior Variance vs c_Q')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/validation_posterior_scaling.png', dpi=300, bbox_inches='tight')
    print(f"\nOptimal c_Q: {c_vals[optimal_idx]} (smallest gap: {gaps[optimal_idx]:.4f})")
    print("Saved: figures/validation_posterior_scaling.png")
    
    return results


def experiment_3_hutchinson_samples(out_file: Path):
    """Experiment 3: Effect of Hutchinson sample size m."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: HUTCHINSON SAMPLE SIZE")
    print("="*70)
    
    m_values = [50, 100, 200, 500]
    results = []
    
    for m in m_values:
        print(f"\nTesting m={m}...")
        result = run_single_experiment(T=50, c_Q=1.0, lambda_reg=0.01, m_hess=m)
        results.append(result)
        _atomic_append_json(_serialize_result(result), out_file)

        print(f"  Traj bound: {result['traj_bound']:.4f} (gap: {result['traj_gap']:.4f})")
        print(f"  Trace(H_bar): {result['trace_H_bar']:.2f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    m_vals = [r['m'] for r in results]
    gaps = [r['traj_gap'] for r in results]
    traces = [r['trace_H_bar'] for r in results]
    
    axes[0].plot(m_vals, gaps, 'o-', linewidth=2)
    axes[0].set_xlabel('Hutchinson Sample Size m')
    axes[0].set_ylabel('Bound Gap')
    axes[0].set_title('Bound Stability vs Sample Size')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(m_vals, traces, 's-', color='orange', linewidth=2)
    axes[1].set_xlabel('Hutchinson Sample Size m')
    axes[1].set_ylabel('Trace(H̄_T)')
    axes[1].set_title('Curvature Estimate vs Sample Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/validation_hutchinson_samples.png', dpi=300, bbox_inches='tight')
    print("\nSaved: figures/validation_hutchinson_samples.png")
    
    return results


def main():
    """Run all validation experiments and save incremental results to disk."""
    parser = argparse.ArgumentParser(description='Run validation experiments and save incremental results')
    parser.add_argument('--out', type=str, default=None, help='Output JSON file to append results to')
    args = parser.parse_args()

    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    if args.out is None:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_file = Path('results') / f'validation_results_{timestamp}.json'
    else:
        out_file = Path(args.out)

    print("\n" + "="*70)
    print("VALIDATION EXPERIMENTS - DIAGNOSING BOUND BEHAVIOR")
    print("="*70)

    # Run experiments (each saves incrementally to out_file)
    experiment_1_trajectory_length(out_file)
    experiment_2_posterior_scaling(out_file)
    experiment_3_hutchinson_samples(out_file)

    print("\n" + "="*70)
    print("ALL VALIDATION EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults are appended to: {out_file}")
    print("\nGenerated figures:")
    print("  - figures/validation_trajectory_length.png")
    print("  - figures/validation_posterior_scaling.png")
    print("  - figures/validation_hutchinson_samples.png")


if __name__ == "__main__":
    main()
