"""
Parallel Validation Experiments
================================

Parallelized version for running many configurations overnight.
Uses multiprocessing to run experiments in parallel across available CPUs.

Usage:
    # Run with all CPUs
    python experiments/parallel_validation.py --out results/parallel_overnight.json
    
    # Run with specific number of workers
    python experiments/parallel_validation.py --out results/parallel.json --workers 4
    
    # Custom configuration sweep
    python experiments/parallel_validation.py --out results/custom.json --T 20,50,100 --c_Q 0.5,1.0,2.0
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
from functools import partial

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
    """Append a single item to a JSON list stored at out_path atomically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(out_path.parent), suffix='.json')
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


def run_single_config(config_tuple):
    """
    Run a single configuration.
    
    Parameters:
    -----------
    config_tuple : tuple
        (T, c_Q, lambda_reg, m_hess, seed, out_file, worker_id)
    
    Returns:
    --------
    result : dict
        Experiment results
    """
    T, c_Q, lambda_reg, m_hess, seed, out_file, worker_id = config_tuple
    
    try:
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
        
        record_every = max(1, T // 20)
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
        
        # Compute bounds
        mu_P = torch.zeros(d)
        Sigma_P = torch.ones(d)
        w_Q = trajectory[-1]
        
        train_risk = trainer.compute_empirical_risk(train_loader)
        test_risk = trainer.compute_empirical_risk(test_loader)
        
        # Trajectory bound
        pac_bayes_traj = GaussianPACBayesBound(
            mu_P=mu_P, Sigma_P=Sigma_P, mu_Q=w_Q,
            Sigma_Q=torch.from_numpy(Sigma_Q).float(),
            r=10, device='cpu', use_diagonal=True
        )
        
        traj_kl = pac_bayes_traj.compute_gaussian_kl()
        n = len(train_loader.dataset)
        delta = 0.05
        traj_bound_val = train_risk + np.sqrt((traj_kl + np.log(1/delta)) / (2*n))
        
        # Baseline bound
        Sigma_baseline = torch.ones(d)
        pac_bayes_base = GaussianPACBayesBound(
            mu_P=mu_P, Sigma_P=Sigma_P, mu_Q=w_Q,
            Sigma_Q=Sigma_baseline,
            r=10, device='cpu', use_diagonal=True
        )
        
        base_kl = pac_bayes_base.compute_gaussian_kl()
        base_bound_val = train_risk + np.sqrt((base_kl + np.log(1/delta)) / (2*n))
        
        # Attach metadata
        result = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'worker_id': int(worker_id),
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
            'final_val_acc': float(trainer.val_accs[-1]),
            'status': 'success'
        }
        
        # Save immediately
        _atomic_append_json(_serialize_result(result), Path(out_file))
        
        print(f"[Worker {worker_id}] ✓ Completed T={T}, c_Q={c_Q}, λ={lambda_reg}, m={m_hess} "
              f"at {datetime.now().strftime('%H:%M:%S')} | Gap: {result['traj_gap']:.4f}")
        
        return result
        
    except Exception as e:
        error_result = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'worker_id': int(worker_id),
            'T': int(T),
            'c_Q': float(c_Q),
            'lambda': float(lambda_reg),
            'm': int(m_hess),
            'seed': int(seed),
            'status': 'error',
            'error_message': str(e)
        }
        
        _atomic_append_json(_serialize_result(error_result), Path(out_file))
        
        print(f"[Worker {worker_id}] ✗ Error T={T}, c_Q={c_Q}: {e}")
        return error_result


def main():
    parser = argparse.ArgumentParser(
        description='Run validation experiments in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all CPU cores
  python experiments/parallel_validation.py --out results/overnight.json
  
  # Run with 4 workers
  python experiments/parallel_validation.py --out results/test.json --workers 4
  
  # Custom configuration sweep
  python experiments/parallel_validation.py --out results/custom.json \\
      --T 20,50,100,200 --c_Q 0.1,0.5,1.0,2.0,5.0 --m 50,100,200,500
        """
    )
    
    parser.add_argument('--out', type=str, required=True, help='Output JSON file')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of parallel workers (default: all CPUs)')
    parser.add_argument('--T', type=str, default='20,50,100,200',
                       help='Comma-separated trajectory lengths (default: 20,50,100,200)')
    parser.add_argument('--c_Q', type=str, default='0.1,0.5,1.0,2.0,5.0',
                       help='Comma-separated c_Q values (default: 0.1,0.5,1.0,2.0,5.0)')
    parser.add_argument('--lambda', dest='lambda_reg', type=str, default='0.01',
                       help='Comma-separated lambda values (default: 0.01)')
    parser.add_argument('--m', type=str, default='50,100,200,500',
                       help='Comma-separated Hutchinson sample sizes (default: 50,100,200,500)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    
    args = parser.parse_args()
    
    # Parse configuration lists
    T_values = [int(x) for x in args.T.split(',')]
    c_Q_values = [float(x) for x in args.c_Q.split(',')]
    lambda_values = [float(x) for x in args.lambda_reg.split(',')]
    m_values = [int(x) for x in args.m.split(',')]
    
    # Create all combinations
    configs = list(product(T_values, c_Q_values, lambda_values, m_values))
    n_configs = len(configs)
    
    # Determine number of workers
    n_workers = args.workers if args.workers else cpu_count()
    n_workers = min(n_workers, n_configs)  # Don't use more workers than configs
    
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare config tuples with worker IDs
    config_tuples = [
        (T, c_Q, lambda_reg, m, args.seed, str(out_file), i % n_workers)
        for i, (T, c_Q, lambda_reg, m) in enumerate(configs)
    ]
    
    start_time = datetime.now()
    print("\n" + "="*70)
    print("PARALLEL VALIDATION EXPERIMENTS")
    print("="*70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {out_file}")
    print(f"Workers: {n_workers} (of {cpu_count()} CPUs)")
    print(f"Total configurations: {n_configs}")
    print(f"  T values: {T_values}")
    print(f"  c_Q values: {c_Q_values}")
    print(f"  λ values: {lambda_values}")
    print(f"  m values: {m_values}")
    print("="*70)
    
    # Run in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_config, config_tuples)
    
    # Count successes and failures
    n_success = sum(1 for r in results if r.get('status') == 'success')
    n_error = sum(1 for r in results if r.get('status') == 'error')
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PARALLEL VALIDATION COMPLETE")
    print("="*70)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Success: {n_success}/{n_configs}")
    print(f"Errors:  {n_error}/{n_configs}")
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
