"""
Visualization and plotting utilities for PAC-Bayes experiments.
===============================================================

This module provides functions to generate publication-quality plots
for the project report.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Set publication-quality style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_training_curves(results: List[Dict], save_path: str = None):
    """
    Plot training and validation curves with error bars.
    
    Parameters:
    -----------
    results : List[Dict]
        List of experiment results
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    
    for result in results:
        hist = result['training_history']
        all_train_losses.append(hist['train_losses'])
        all_val_losses.append(hist['val_losses'])
        all_train_accs.append(hist['train_accs'])
        all_val_accs.append(hist['val_accs'])
    
    # Convert to arrays
    train_losses = np.array(all_train_losses)
    val_losses = np.array(all_val_losses)
    train_accs = np.array(all_train_accs)
    val_accs = np.array(all_val_accs)
    
    epochs = np.arange(1, len(train_losses[0]) + 1)
    
    # Plot losses
    ax = axes[0]
    ax.plot(epochs, train_losses.mean(axis=0), label='Train', linewidth=2)
    ax.fill_between(
        epochs,
        train_losses.mean(axis=0) - train_losses.std(axis=0),
        train_losses.mean(axis=0) + train_losses.std(axis=0),
        alpha=0.3
    )
    
    ax.plot(epochs, val_losses.mean(axis=0), label='Validation', linewidth=2)
    ax.fill_between(
        epochs,
        val_losses.mean(axis=0) - val_losses.std(axis=0),
        val_losses.mean(axis=0) + val_losses.std(axis=0),
        alpha=0.3
    )
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax = axes[1]
    ax.plot(epochs, train_accs.mean(axis=0), label='Train', linewidth=2)
    ax.fill_between(
        epochs,
        train_accs.mean(axis=0) - train_accs.std(axis=0),
        train_accs.mean(axis=0) + train_accs.std(axis=0),
        alpha=0.3
    )
    
    ax.plot(epochs, val_accs.mean(axis=0), label='Validation', linewidth=2)
    ax.fill_between(
        epochs,
        val_accs.mean(axis=0) - val_accs.std(axis=0),
        val_accs.mean(axis=0) + val_accs.std(axis=0),
        alpha=0.3
    )
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_bound_comparison(results: List[Dict], save_path: str = None):
    """
    Plot comparison of bounds vs actual test risk.
    
    Parameters:
    -----------
    results : List[Dict]
        List of experiment results
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    test_risks = [r['test_risk'] for r in results]
    traj_bounds = [r['trajectory_bound']['pac_bayes_bound'] for r in results]
    base_bounds = [r['baseline_bound']['pac_bayes_bound'] for r in results]
    train_risks = [r['train_risk'] for r in results]
    
    # Create bar plot
    x = np.arange(4)
    means = [
        np.mean(train_risks),
        np.mean(test_risks),
        np.mean(traj_bounds),
        np.mean(base_bounds)
    ]
    stds = [
        np.std(train_risks),
        np.std(test_risks),
        np.std(traj_bounds),
        np.std(base_bounds)
    ]
    labels = ['Train Risk', 'Test Risk', 'Trajectory-Aware\nBound', 'Baseline\nBound']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Risk / Bound Value')
    ax.set_title('PAC-Bayes Bounds vs Empirical Risk')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved bound comparison to {save_path}")
    
    return fig


def plot_kl_decomposition(results: List[Dict], save_path: str = None):
    """
    Plot KL divergence decomposition.
    
    Parameters:
    -----------
    results : List[Dict]
        List of experiment results
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract KL components (average over seeds)
    kl_decomp = results[0]['kl_decomposition']
    
    # Get relevant components
    if 'trace_full' in kl_decomp:
        # Full matrix decomposition
        components = {
            'Top Subspace\nAlignment': np.mean([r['kl_decomposition']['top_subspace_alignment'] for r in results]),
            'Complement\nAlignment': np.mean([r['kl_decomposition']['complement_alignment'] for r in results]),
            'Trace\nTop Subspace': np.mean([r['kl_decomposition']['trace_top_subspace'] for r in results]),
            'Trace\nResidual': np.mean([r['kl_decomposition']['trace_residual'] for r in results]),
            'Log-Det\nTerm': np.mean([r['kl_decomposition']['log_det_term'] for r in results]),
            'Dimension\nTerm': np.mean([r['kl_decomposition']['dimension_term'] for r in results])
        }
    else:
        # Diagonal approximation
        components = {
            'Mean\nAlignment': np.mean([r['kl_decomposition']['mean_alignment'] for r in results]),
            'Trace\nTerm': np.mean([r['kl_decomposition']['trace_term'] for r in results]),
            'Log-Det\nTerm': np.mean([r['kl_decomposition']['log_det_term'] for r in results]),
            'Dimension\nTerm': np.mean([r['kl_decomposition']['dimension_term'] for r in results])
        }
    
    labels = list(components.keys())
    values = list(components.values())
    
    colors = sns.color_palette("husl", len(labels))
    bars = ax.barh(labels, values, color=colors, edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Contribution to KL Divergence')
    ax.set_title('KL Divergence Decomposition')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}',
                ha='left' if width >= 0 else 'right',
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved KL decomposition to {save_path}")
    
    return fig


def create_results_table(results: List[Dict], save_path: str = None) -> pd.DataFrame:
    """
    Create summary table of results.
    
    Parameters:
    -----------
    results : List[Dict]
        List of experiment results
    save_path : str
        Path to save table (CSV or LaTeX)
        
    Returns:
    --------
    df : pd.DataFrame
        Results table
    """
    data = {
        'Metric': [
            'Number of Parameters',
            'Training Samples',
            'Train Risk',
            'Validation Risk',
            'Test Risk',
            'KL Divergence',
            'Trajectory-Aware Bound',
            'Baseline Bound',
            'Bound Gap (Traj - Test)',
            'Bound Gap (Base - Test)'
        ],
        'Mean': [
            results[0]['n_params'],
            results[0]['n_train'],
            np.mean([r['train_risk'] for r in results]),
            np.mean([r['val_risk'] for r in results]),
            np.mean([r['test_risk'] for r in results]),
            np.mean([r['trajectory_bound']['kl_divergence'] for r in results]),
            np.mean([r['trajectory_bound']['pac_bayes_bound'] for r in results]),
            np.mean([r['baseline_bound']['pac_bayes_bound'] for r in results]),
            np.mean([r['trajectory_bound']['pac_bayes_bound'] - r['test_risk'] for r in results]),
            np.mean([r['baseline_bound']['pac_bayes_bound'] - r['test_risk'] for r in results])
        ],
        'Std': [
            0,
            0,
            np.std([r['train_risk'] for r in results]),
            np.std([r['val_risk'] for r in results]),
            np.std([r['test_risk'] for r in results]),
            np.std([r['trajectory_bound']['kl_divergence'] for r in results]),
            np.std([r['trajectory_bound']['pac_bayes_bound'] for r in results]),
            np.std([r['baseline_bound']['pac_bayes_bound'] for r in results]),
            np.std([r['trajectory_bound']['pac_bayes_bound'] - r['test_risk'] for r in results]),
            np.std([r['baseline_bound']['pac_bayes_bound'] - r['test_risk'] for r in results])
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Format numbers
    df['Value'] = df.apply(
        lambda row: f"{row['Mean']:.4f} ± {row['Std']:.4f}" if row['Std'] > 0 else f"{int(row['Mean'])}", 
        axis=1
    )
    
    df_display = df[['Metric', 'Value']]
    
    if save_path:
        if save_path.endswith('.csv'):
            df_display.to_csv(save_path, index=False)
        elif save_path.endswith('.tex'):
            latex_str = df_display.to_latex(index=False, escape=False)
            with open(save_path, 'w') as f:
                f.write(latex_str)
        print(f"Saved results table to {save_path}")
    
    return df_display


def generate_all_figures(results_path: str, output_dir: str = "figures"):
    """
    Generate all figures from results file.
    
    Parameters:
    -----------
    results_path : str
        Path to results JSON file
    output_dir : str
        Directory to save figures
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating figures from {results_path}...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate plots
    plot_training_curves(results, save_path=f"{output_dir}/training_curves.png")
    plot_bound_comparison(results, save_path=f"{output_dir}/bound_comparison.png")
    plot_kl_decomposition(results, save_path=f"{output_dir}/kl_decomposition.png")
    
    # Generate table
    create_results_table(results, save_path=f"{output_dir}/results_table.csv")
    create_results_table(results, save_path=f"{output_dir}/results_table.tex")
    
    print(f"\nAll figures generated successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
        generate_all_figures(results_path, output_dir)
    else:
        print("Usage: python experiments/visualize_results.py <results_json_path> [output_dir]")
