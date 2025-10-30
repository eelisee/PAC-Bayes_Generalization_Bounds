#!/usr/bin/env python3
"""
Detailed correlation and parameter influence analysis.
Answers the key questions about trajectory-aware PAC-Bayes bounds.
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_results(filepath):
    """Load and filter successful results."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    results = [r for r in results if r.get('status') == 'success']
    return pd.DataFrame(results)

def analyze_parameter_correlations(df):
    """Analyze correlations between parameters and outcomes."""
    print("=" * 80)
    print("PARAMETER CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Select relevant columns
    param_cols = ['T', 'c_Q', 'lambda', 'm']
    outcome_cols = ['traj_gap', 'base_gap', 'improvement_percent', 'traj_kl', 
                   'test_risk', 'trace_H_bar', 'trace_Sigma_Q']
    
    # Compute correlation matrix
    corr_data = df[param_cols + outcome_cols]
    corr_matrix = corr_data.corr()
    
    # Focus on parameter-outcome correlations
    param_outcome_corr = corr_matrix.loc[param_cols, outcome_cols]
    
    print("\nCorrelation between Parameters and Outcomes:")
    print(param_outcome_corr.round(3))
    
    # Statistical significance tests
    print("\n" + "-" * 80)
    print("Statistical Significance (Spearman correlation, p-values):")
    print("-" * 80)
    
    for param in param_cols:
        print(f"\n{param}:")
        for outcome in outcome_cols:
            rho, pval = stats.spearmanr(df[param], df[outcome])
            sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  vs {outcome:20s}: ρ={rho:+.3f}, p={pval:.4f} {sig_marker}")
    
    return param_outcome_corr

def analyze_trajectory_length_effect(df):
    """Q1: How does trajectory length T affect bound tightness?"""
    print("\n" + "=" * 80)
    print("Q1: TRAJECTORY LENGTH EFFECT")
    print("=" * 80)
    
    T_analysis = df.groupby('T').agg({
        'traj_gap': ['mean', 'std', 'min', 'max'],
        'base_gap': ['mean', 'std'],
        'improvement_percent': ['mean', 'std'],
        'traj_kl': ['mean', 'std'],
        'trace_H_bar': ['mean', 'std']
    }).round(4)
    
    print("\nTrajectory Gap by T:")
    print(T_analysis['traj_gap'])
    
    print("\nImprovement by T:")
    print(T_analysis['improvement_percent'])
    
    # Statistical test: monotonic trend
    rho, pval = stats.spearmanr(df['T'], df['traj_gap'])
    print(f"\nSpearman correlation (T vs traj_gap): ρ={rho:.3f}, p={pval:.4f}")
    
    if pval < 0.05:
        if rho > 0:
            print("✓ Significant POSITIVE correlation: Longer trajectories → WORSE bounds")
        else:
            print("✓ Significant NEGATIVE correlation: Longer trajectories → BETTER bounds")
    else:
        print("✗ No significant correlation between T and bound tightness")
    
    # ANOVA test
    T_groups = [df[df['T'] == T]['traj_gap'].values for T in sorted(df['T'].unique())]
    f_stat, p_anova = stats.f_oneway(*T_groups)
    print(f"\nANOVA test across T values: F={f_stat:.2f}, p={p_anova:.4f}")
    if p_anova < 0.05:
        print("✓ Significant difference in bound tightness across T values")
    else:
        print("✗ No significant difference across T values")
    
    return T_analysis

def analyze_posterior_scaling_effect(df):
    """Q2: What is the optimal posterior scaling c_Q?"""
    print("\n" + "=" * 80)
    print("Q2: OPTIMAL POSTERIOR SCALING")
    print("=" * 80)
    
    cQ_analysis = df.groupby('c_Q').agg({
        'traj_gap': ['mean', 'std', 'min', 'max'],
        'traj_kl': ['mean', 'std'],
        'trace_Sigma_Q': ['mean', 'std'],
        'improvement_percent': ['mean', 'std']
    }).round(4)
    
    print("\nTrajectory Gap by c_Q:")
    print(cQ_analysis['traj_gap'])
    
    # Find optimal c_Q
    optimal_cQ = cQ_analysis[('traj_gap', 'mean')].idxmin()
    optimal_gap = cQ_analysis.loc[optimal_cQ, ('traj_gap', 'mean')]
    
    print(f"\nOptimal c_Q: {optimal_cQ}")
    print(f"  Mean gap: {optimal_gap:.4f}")
    print(f"  Std gap: {cQ_analysis.loc[optimal_cQ, ('traj_gap', 'std')]:.4f}")
    print(f"  Mean KL: {cQ_analysis.loc[optimal_cQ, ('traj_kl', 'mean')]:.4f}")
    
    # Test for non-linear relationship (polynomial fit)
    cQ_vals = df.groupby('c_Q')['traj_gap'].mean()
    x = np.log10(cQ_vals.index)
    y = np.log10(cQ_vals.values)
    
    # Linear fit in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"\nLog-log relationship: slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.4f}")
    
    if abs(slope) > 0.5 and p_value < 0.05:
        print(f"✓ Strong power-law relationship: traj_gap ∝ c_Q^{slope:.2f}")
    
    # KL vs Gap trade-off
    print("\nKL-Gap Trade-off Analysis:")
    for cQ in sorted(df['c_Q'].unique()):
        subset = df[df['c_Q'] == cQ]
        print(f"  c_Q={cQ:5.2f}: gap={subset['traj_gap'].mean():8.4f}, KL={subset['traj_kl'].mean():8.4f}")
    
    return cQ_analysis

def analyze_regularization_effect(df):
    """Q3: How does regularization λ affect the bounds?"""
    print("\n" + "=" * 80)
    print("Q3: REGULARIZATION EFFECT")
    print("=" * 80)
    
    lambda_analysis = df.groupby('lambda').agg({
        'traj_gap': ['mean', 'std', 'min'],
        'trace_H_bar': ['mean', 'std'],
        'test_risk': ['mean', 'std'],
        'improvement_percent': ['mean', 'std']
    }).round(4)
    
    print("\nTrajectory Gap by λ:")
    print(lambda_analysis['traj_gap'])
    
    print("\nCurvature (Trace H̄) by λ:")
    print(lambda_analysis['trace_H_bar'])
    
    print("\nTest Risk by λ:")
    print(lambda_analysis['test_risk'])
    
    # Correlation analysis
    rho_gap, p_gap = stats.spearmanr(df['lambda'], df['traj_gap'])
    rho_curv, p_curv = stats.spearmanr(df['lambda'], df['trace_H_bar'])
    
    print(f"\nλ vs traj_gap: ρ={rho_gap:+.3f}, p={p_gap:.4f}")
    print(f"λ vs trace_H_bar: ρ={rho_curv:+.3f}, p={p_curv:.4f}")
    
    if p_gap < 0.05:
        if rho_gap < 0:
            print("✓ Stronger regularization → BETTER bounds")
        else:
            print("✗ Stronger regularization → WORSE bounds")
    
    return lambda_analysis

def analyze_hutchinson_effect(df):
    """Q4: Does increasing Hutchinson samples improve stability?"""
    print("\n" + "=" * 80)
    print("Q4: HUTCHINSON SAMPLE SIZE EFFECT")
    print("=" * 80)
    
    m_analysis = df.groupby('m').agg({
        'traj_gap': ['mean', 'std', 'min', 'max'],
        'trace_H_bar': ['mean', 'std'],
        'improvement_percent': ['mean', 'std']
    }).round(4)
    
    print("\nTrajectory Gap by m (sample size):")
    print(m_analysis['traj_gap'])
    
    print("\nVariability Analysis (Std Dev of Gap):")
    for m in sorted(df['m'].unique()):
        std = m_analysis.loc[m, ('traj_gap', 'std')]
        print(f"  m={m:4d}: std={std:.4f}")
    
    # Test if std decreases with m
    m_vals = sorted(df['m'].unique())
    stds = [m_analysis.loc[m, ('traj_gap', 'std')] for m in m_vals]
    rho, pval = stats.spearmanr(m_vals, stds)
    
    print(f"\nCorrelation (m vs std): ρ={rho:+.3f}, p={pval:.4f}")
    
    if pval < 0.05 and rho < 0:
        print("✓ More samples → LOWER variance (better stability)")
    elif pval < 0.05 and rho > 0:
        print("✗ More samples → HIGHER variance (unexpected!)")
    else:
        print("~ No significant effect on stability")
    
    return m_analysis

def analyze_interaction_effects(df):
    """Q5: Are there interaction effects between parameters?"""
    print("\n" + "=" * 80)
    print("Q5: PARAMETER INTERACTION EFFECTS")
    print("=" * 80)
    
    # T × c_Q interaction
    print("\nInteraction: T × c_Q")
    interaction_TcQ = df.groupby(['T', 'c_Q'])['traj_gap'].mean().unstack()
    print(interaction_TcQ.round(4))
    
    # Find best combination
    best_idx = df['traj_gap'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest combination overall:")
    print(f"  T={best['T']}, c_Q={best['c_Q']}, λ={best['lambda']}, m={best['m']}")
    print(f"  Trajectory gap: {best['traj_gap']:.4f}")
    print(f"  Improvement: {best['improvement_percent']:.2f}%")
    
    # T × λ interaction
    print("\n\nInteraction: T × λ")
    interaction_Tlambda = df.groupby(['T', 'lambda'])['traj_gap'].mean().unstack()
    print(interaction_Tlambda.round(4))
    
    # c_Q × λ interaction
    print("\n\nInteraction: c_Q × λ")
    interaction_cQlambda = df.groupby(['c_Q', 'lambda'])['traj_gap'].mean().unstack()
    print(interaction_cQlambda.round(4))
    
    return interaction_TcQ

def why_trajectory_worse(df):
    """Diagnostic: Why is trajectory bound worse than baseline?"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: WHY IS TRAJECTORY BOUND WORSE?")
    print("=" * 80)
    
    # Compare KL decomposition
    print("\nKL Divergence Comparison:")
    print(f"  Trajectory mean KL: {df['traj_kl'].mean():.4f} ± {df['traj_kl'].std():.4f}")
    print(f"  Baseline mean KL: {df['base_kl'].mean():.4f} ± {df['base_kl'].std():.4f}")
    print(f"  Ratio (traj/base): {(df['traj_kl']/df['base_kl']).mean():.2f}×")
    
    # Curvature analysis
    print("\nCurvature Statistics:")
    print(f"  Trace(H̄_T) mean: {df['trace_H_bar'].mean():.4f}")
    print(f"  Trace(Σ_Q) mean: {df['trace_Sigma_Q'].mean():.4f}")
    print(f"  Product mean: {(df['trace_H_bar'] * df['trace_Sigma_Q']).mean():.4f}")
    
    # Check if KL dominates
    df['kl_ratio'] = df['traj_kl'] / df['base_kl']
    df['gap_ratio'] = df['traj_gap'] / df['base_gap']
    
    print("\nRatio Analysis:")
    print(f"  Mean KL ratio: {df['kl_ratio'].mean():.2f}×")
    print(f"  Mean gap ratio: {df['gap_ratio'].mean():.2f}×")
    
    # Correlation between KL and gap deterioration
    rho, pval = stats.spearmanr(df['kl_ratio'], df['gap_ratio'])
    print(f"  Correlation (KL_ratio vs gap_ratio): ρ={rho:.3f}, p={pval:.4f}")
    
    if rho > 0.5 and pval < 0.05:
        print("\n✓ DIAGNOSIS: Increased KL divergence is the PRIMARY cause of worse bounds")
        print("  → Trajectory posterior is too different from prior")
        print("  → Need to adjust prior or posterior scaling")
    
    # Check for extreme cases
    extreme_cases = df.nlargest(5, 'gap_ratio')[['T', 'c_Q', 'lambda', 'm', 
                                                   'traj_gap', 'base_gap', 'gap_ratio', 'kl_ratio']]
    print("\nWorst 5 configurations (highest gap_ratio):")
    print(extreme_cases.round(2))
    
    return df[['kl_ratio', 'gap_ratio']]

def generate_advanced_figures(df, output_dir):
    """Generate advanced correlation and interaction plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 150
    
    # Figure 1: Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    param_cols = ['T', 'c_Q', 'lambda', 'm']
    outcome_cols = ['traj_gap', 'base_gap', 'improvement_percent', 'traj_kl', 'test_risk']
    corr_matrix = df[param_cols + outcome_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Parameter-Outcome Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'correlation_heatmap.png'}")
    
    # Figure 2: T × c_Q interaction contour
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot = df.groupby(['T', 'c_Q'])['traj_gap'].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Trajectory Gap'})
    ax.set_xlabel('Posterior Scaling c_Q', fontsize=12)
    ax.set_ylabel('Trajectory Length T', fontsize=12)
    ax.set_title('T × c_Q Interaction Effect on Bound Tightness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'interaction_T_cQ.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'interaction_T_cQ.png'}")
    
    # Figure 3: Multi-panel parameter effects
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # T effect
    T_grouped = df.groupby('T')['traj_gap'].agg(['mean', 'std']).reset_index()
    axes[0, 0].errorbar(T_grouped['T'], T_grouped['mean'], yerr=T_grouped['std'],
                       marker='o', linewidth=2, capsize=5, markersize=8)
    axes[0, 0].set_xlabel('Trajectory Length T', fontsize=11)
    axes[0, 0].set_ylabel('Trajectory Gap', fontsize=11)
    axes[0, 0].set_title('Effect of Trajectory Length', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # c_Q effect (log scale)
    cQ_grouped = df.groupby('c_Q')['traj_gap'].agg(['mean', 'std']).reset_index()
    axes[0, 1].errorbar(cQ_grouped['c_Q'], cQ_grouped['mean'], yerr=cQ_grouped['std'],
                       marker='s', linewidth=2, capsize=5, markersize=8, color='orange')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Posterior Scaling c_Q (log)', fontsize=11)
    axes[0, 1].set_ylabel('Trajectory Gap (log)', fontsize=11)
    axes[0, 1].set_title('Effect of Posterior Scaling', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # λ effect
    lambda_grouped = df.groupby('lambda')['traj_gap'].agg(['mean', 'std']).reset_index()
    axes[1, 0].errorbar(lambda_grouped['lambda'], lambda_grouped['mean'], 
                       yerr=lambda_grouped['std'],
                       marker='^', linewidth=2, capsize=5, markersize=8, color='green')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Regularization λ (log)', fontsize=11)
    axes[1, 0].set_ylabel('Trajectory Gap', fontsize=11)
    axes[1, 0].set_title('Effect of Regularization', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # m effect
    m_grouped = df.groupby('m')['traj_gap'].agg(['mean', 'std']).reset_index()
    axes[1, 1].errorbar(m_grouped['m'], m_grouped['mean'], yerr=m_grouped['std'],
                       marker='D', linewidth=2, capsize=5, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Hutchinson Samples m', fontsize=11)
    axes[1, 1].set_ylabel('Trajectory Gap', fontsize=11)
    axes[1, 1].set_title('Effect of Sample Size', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'parameter_effects.png'}")
    
    # Figure 4: KL vs Gap diagnostic
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(df['traj_kl'], df['traj_gap'], alpha=0.3, c=df['c_Q'], 
                   cmap='viridis', s=20)
    axes[0].set_xlabel('Trajectory KL Divergence', fontsize=11)
    axes[0].set_ylabel('Trajectory Gap', fontsize=11)
    axes[0].set_title('KL vs Gap (colored by c_Q)', fontsize=12, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_label('c_Q', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    df['kl_ratio'] = df['traj_kl'] / df['base_kl']
    df['gap_ratio'] = df['traj_gap'] / df['base_gap']
    axes[1].scatter(df['kl_ratio'], df['gap_ratio'], alpha=0.3, c=df['T'],
                   cmap='plasma', s=20)
    axes[1].axline((1, 1), slope=1, color='red', linestyle='--', linewidth=2, label='y=x')
    axes[1].set_xlabel('KL Ratio (traj/base)', fontsize=11)
    axes[1].set_ylabel('Gap Ratio (traj/base)', fontsize=11)
    axes[1].set_title('Ratio Analysis (colored by T)', fontsize=12, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('T', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_kl_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'diagnostic_kl_gap.png'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/detailed_correlation_analysis.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    print(f"Loading results from: {results_file}\n")
    
    df = load_results(results_file)
    print(f"✓ Loaded {len(df)} successful experiments\n")
    
    # Run all analyses
    analyze_parameter_correlations(df)
    analyze_trajectory_length_effect(df)
    analyze_posterior_scaling_effect(df)
    analyze_regularization_effect(df)
    analyze_hutchinson_effect(df)
    analyze_interaction_effects(df)
    why_trajectory_worse(df)
    
    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING ADVANCED FIGURES")
    print("=" * 80)
    output_dir = Path('results/analysis')
    generate_advanced_figures(df, output_dir)
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll outputs saved to: results/analysis/")

if __name__ == '__main__':
    main()
