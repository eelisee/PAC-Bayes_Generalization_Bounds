"""
Simple smoke test to verify the PAC-Bayes implementation works.
This runs a minimal experiment to check all components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

print("="*60)
print("PAC-Bayes Implementation Smoke Test")
print("="*60)

# Test 1: Data loading
print("\n1. Testing data loader...")
try:
    from src.data_loader import get_mnist_binary
    train_loader, val_loader, test_loader, input_dim = get_mnist_binary(
        data_dir='../data',
        batch_size=64,
        flatten=True,
        seed=42
    )
    print(f"   ✓ Data loaded: {len(train_loader.dataset)} train samples, dim={input_dim}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Model creation
print("\n2. Testing model creation...")
try:
    from src.models import create_model, count_parameters
    model = create_model('logistic', input_dim)
    n_params = count_parameters(model)
    print(f"   ✓ Model created: {n_params} parameters")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Training with trajectory
print("\n3. Testing training with trajectory...")
try:
    from src.training import train_model_with_trajectory
    from src.utils import set_random_seed
    
    set_random_seed(42)
    device = 'cpu'  # Use CPU for testing
    
    # Quick training (just 2 epochs)
    model, history = train_model_with_trajectory(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=0.01,
        device=device,
        record_every=1,
        verbose=False
    )
    
    print(f"   ✓ Training complete: {len(history['trajectory'])} trajectory points")
    print(f"   ✓ Final train loss: {history['train_losses'][-1]:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Hessian estimation
print("\n4. Testing Hessian estimation...")
try:
    from src.hessian_estimator import TrajectoryHessianEstimator
    
    # Use small subset for fast testing
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_loader.dataset, range(100)),
        batch_size=50
    )
    
    hess_estimator = TrajectoryHessianEstimator(
        model=model,
        trajectory=history['trajectory'],
        data_loader=subset_loader,
        criterion=torch.nn.BCELoss(),
        device=device,
        lambda_reg=0.01,
        c_Q=1.0,
        use_diagonal=True,
        num_hess_samples=10,  # Small for testing
        verbose=False
    )
    
    mu_Q, Sigma_Q = hess_estimator.get_posterior_params()
    print(f"   ✓ Hessian estimated: mu_Q shape={mu_Q.shape}, Sigma_Q shape={Sigma_Q.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: PAC-Bayes bound computation
print("\n5. Testing PAC-Bayes bound computation...")
try:
    from src.pac_bayes_bounds import GaussianPACBayesBound, compute_baseline_pac_bayes_bound
    from src.training import TrajectoryTrainer
    
    # Define prior
    mu_P = torch.zeros_like(mu_Q)
    Sigma_P = torch.ones_like(Sigma_Q)
    
    # Compute empirical risk
    trainer = TrajectoryTrainer(
        model=model,
        optimizer=None,
        criterion=torch.nn.BCELoss(),
        device=device,
        verbose=False
    )
    
    train_risk = trainer.compute_empirical_risk(train_loader)
    
    # Compute bound
    pac_computer = GaussianPACBayesBound(
        mu_P=mu_P,
        Sigma_P=Sigma_P,
        mu_Q=mu_Q,
        Sigma_Q=Sigma_Q,
        r=10,
        device=device,
        use_diagonal=True
    )
    
    bound_info = pac_computer.compute_pac_bayes_bound(
        empirical_risk=train_risk,
        n_samples=len(train_loader.dataset),
        delta=0.05
    )
    
    print(f"   ✓ PAC-Bayes bound computed: {bound_info['pac_bayes_bound']:.4f}")
    print(f"   ✓ Train risk: {train_risk:.4f}")
    print(f"   ✓ KL divergence: {bound_info['kl_divergence']:.4f}")
    
    # Baseline bound
    baseline = compute_baseline_pac_bayes_bound(
        empirical_risk=train_risk,
        n_samples=len(train_loader.dataset),
        d=n_params,
        delta=0.05
    )
    print(f"   ✓ Baseline bound: {baseline['pac_bayes_bound']:.4f}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("SMOKE TEST PASSED ✓")
print("="*60)
print("\nAll components are working correctly!")
print("You can now run full experiments with:")
print("  python experiments/run_experiment.py")
print("="*60)
