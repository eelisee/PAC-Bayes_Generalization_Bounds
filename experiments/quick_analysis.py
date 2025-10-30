#!/usr/bin/env python3
"""
Quick analysis script for interrupted unified experiment runs.
Usage: python experiments/quick_analysis.py results/unified_extensive_*.json
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(filepath):
    """Load and filter successful results."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Filter successful runs
    results = [r for r in results if r.get('status') == 'success']
    return pd.DataFrame(results)

def print_summary(df):
    """Print overall statistics."""
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal successful experiments: {len(df)}")
    print(f"\nConfiguration space:")
    print(f"  T values: {sorted(df['T'].unique())}")
    print(f"  c_Q values: {sorted(df['c_Q'].unique())}")
    print(f"  λ values: {sorted(df['lambda'].unique())}")
    print(f"  m values: {sorted(df['m'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    
    print("\n" + "=" * 70)
    print("KEY STATISTICS")
    print("=" * 70)
    print(f"\nTest Risk:")
    print(f"  Mean: {df['test_risk'].mean():.4f} ± {df['test_risk'].std():.4f}")
    print(f"  Range: [{df['test_risk'].min():.4f}, {df['test_risk'].max():.4f}]")
    
    print(f"\nTrajectory Bound Gap:")
    print(f"  Mean: {df['traj_gap'].mean():.4f} ± {df['traj_gap'].std():.4f}")
    print(f"  Range: [{df['traj_gap'].min():.4f}, {df['traj_gap'].max():.4f}]")
    
    print(f"\nBaseline Bound Gap:")
    print(f"  Mean: {df['base_gap'].mean():.4f} ± {df['base_gap'].std():.4f}")
    print(f"  Range: [{df['base_gap'].min():.4f}, {df['base_gap'].max():.4f}]")
    
    print(f"\nImprovement:")
    print(f"  Mean: {df['improvement_percent'].mean():.2f}% ± {df['improvement_percent'].std():.2f}%")
    print(f"  Range: [{df['improvement_percent'].min():.2f}%, {df['improvement_percent'].max():.2f}%]")
    print(f"  Configs with improvement > 0: {(df['improvement_percent'] > 0).sum()}/{len(df)} ({100*(df['improvement_percent'] > 0).mean():.1f}%)")
    
    # Best configuration
    best_idx = df['traj_gap'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest Configuration (lowest trajectory gap):")
    print(f"  T={best['T']}, c_Q={best['c_Q']}, λ={best['lambda']}, m={best['m']}")
    print(f"  Trajectory gap: {best['traj_gap']:.4f}")
    print(f"  Improvement: {best['improvement_percent']:.2f}%")
    print(f"  Test risk: {best['test_risk']:.4f}")

def generate_figures(df, output_dir):
    """Generate all analysis figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 150
    
    # Figure 1: Trajectory Length Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    axes[0].set_xlabel('Trajectory Length T', fontsize=12)
    axes[0].set_ylabel('Bound Gap', fontsize=12)
    axes[0].set_title('Bound Tightness vs Trajectory Length', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].errorbar(T_vals, T_grouped[('improvement_percent', 'mean')],
                    yerr=T_grouped[('improvement_percent', 'std')],
                    marker='D', color='green', linewidth=2, capsize=5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Trajectory Length T', fontsize=12)
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].set_title('Bound Improvement vs T', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    df.boxplot(column='traj_gap', by='T', ax=axes[2])
    axes[2].set_xlabel('Trajectory Length T', fontsize=12)
    axes[2].set_ylabel('Trajectory Bound Gap', fontsize=12)
    axes[2].set_title('Gap Distribution by T', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'trajectory_length_analysis.png'}")
    
    # Figure 2: Posterior Scaling Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cQ_grouped = df.groupby('c_Q').agg({
        'traj_gap': ['mean', 'std'],
        'traj_kl': ['mean', 'std'],
        'trace_Sigma_Q': ['mean', 'std']
    }).reset_index()
    
    cQ_vals = cQ_grouped['c_Q'].values
    
    axes[0].errorbar(cQ_vals, cQ_grouped[('traj_gap', 'mean')],
                    yerr=cQ_grouped[('traj_gap', 'std')],
                    marker='o', linewidth=2, capsize=5)
    optimal_idx = cQ_grouped[('traj_gap', 'mean')].idxmin()
    axes[0].scatter([cQ_vals[optimal_idx]], [cQ_grouped[('traj_gap', 'mean')].iloc[optimal_idx]],
                   s=300, marker='*', color='red', zorder=5, 
                   label=f'Optimal: c_Q={cQ_vals[optimal_idx]:.2f}')
    axes[0].set_xlabel('Posterior Scaling c_Q', fontsize=12)
    axes[0].set_ylabel('Bound Gap', fontsize=12)
    axes[0].set_title('Optimal c_Q Selection', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].errorbar(cQ_vals, cQ_grouped[('traj_kl', 'mean')],
                    yerr=cQ_grouped[('traj_kl', 'std')],
                    marker='s', color='orange', linewidth=2, capsize=5)
    axes[1].set_xlabel('Posterior Scaling c_Q', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('KL vs c_Q', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].errorbar(cQ_vals, cQ_grouped[('trace_Sigma_Q', 'mean')],
                    yerr=cQ_grouped[('trace_Sigma_Q', 'std')],
                    marker='^', color='green', linewidth=2, capsize=5)
    axes[2].set_xlabel('Posterior Scaling c_Q', fontsize=12)
    axes[2].set_ylabel('Trace(Σ_Q)', fontsize=12)
    axes[2].set_title('Posterior Variance vs c_Q', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'posterior_scaling_analysis.png'}")
    
    # Figure 3: Heatmap T vs c_Q
    pivot = df.groupby(['T', 'c_Q'])['improvement_percent'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Improvement (%)'})
    plt.xlabel('Posterior Scaling c_Q', fontsize=12)
    plt.ylabel('Trajectory Length T', fontsize=12)
    plt.title('Bound Improvement Heatmap: T vs c_Q', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_T_cQ.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'heatmap_T_cQ.png'}")
    
    # Figure 4: Overall comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(df['base_gap'], df['traj_gap'], alpha=0.3)
    axes[0, 0].plot([df['base_gap'].min(), df['base_gap'].max()],
                    [df['base_gap'].min(), df['base_gap'].max()],
                    'r--', label='y=x')
    axes[0, 0].set_xlabel('Baseline Gap', fontsize=11)
    axes[0, 0].set_ylabel('Trajectory Gap', fontsize=11)
    axes[0, 0].set_title('Trajectory vs Baseline Gap', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df['improvement_percent'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Improvement (%)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Improvement Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(df['traj_kl'], df['traj_gap'], alpha=0.3, c=df['c_Q'], cmap='viridis')
    axes[1, 0].set_xlabel('KL Divergence', fontsize=11)
    axes[1, 0].set_ylabel('Trajectory Gap', fontsize=11)
    axes[1, 0].set_title('Gap vs KL Divergence', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df['test_risk'], df['traj_gap'], alpha=0.3, c=df['T'], cmap='plasma')
    axes[1, 1].set_xlabel('Test Risk', fontsize=11)
    axes[1, 1].set_ylabel('Trajectory Gap', fontsize=11)
    axes[1, 1].set_title('Gap vs Test Risk', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'overall_comparison.png'}")

def export_csv(df, output_dir):
    """Export results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"✓ Saved: {output_dir / 'all_results.csv'}")
    
    # Summary by hyperparameters
    summary = df.groupby(['T', 'c_Q', 'lambda', 'm']).agg({
        'test_risk': ['mean', 'std'],
        'traj_gap': ['mean', 'std', 'min', 'max'],
        'base_gap': ['mean', 'std'],
        'improvement_percent': ['mean', 'std'],
        'traj_kl': ['mean', 'std']
    }).round(4)
    summary.to_csv(output_dir / 'summary_by_config.csv')
    print(f"✓ Saved: {output_dir / 'summary_by_config.csv'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/quick_analysis.py <results_file.json>")
        print("\nExample:")
        print("  python experiments/quick_analysis.py results/unified_extensive_*.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    print(f"Loading results from: {results_file}")
    
    # Load data
    df = load_results(results_file)
    print(f"✓ Loaded {len(df)} successful experiments\n")
    
    # Print summary
    print_summary(df)
    
    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    output_dir = Path('results/analysis')
    generate_figures(df, output_dir)
    
    # Export CSV
    print("\n" + "=" * 70)
    print("EXPORTING DATA")
    print("=" * 70)
    export_csv(df, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Check the generated PNG figures")
    print("  2. Open all_results.csv in Excel/Jupyter for detailed analysis")
    print("  3. Use the Jupyter notebook for interactive exploration")

if __name__ == '__main__':
    main()
