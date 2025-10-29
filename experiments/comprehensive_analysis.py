"""
Comprehensive analysis module for PAC-Bayes experiments.
========================================================

This module provides detailed analytics to validate theoretical expectations:
- Trajectory analysis (convergence, stability)
- Hessian eigenspectrum analysis
- KL decomposition breakdown
- Hyperparameter sensitivity studies
- Component-wise contributions
- Comparative analysis across configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer for PAC-Bayes experimental results.
    """
    
    def __init__(self, results: List[Dict], output_dir: str = "figures"):
        """
        Initialize analyzer with experimental results.
        
        Parameters:
        -----------
        results : List[Dict]
            List of experiment results (from multiple seeds)
        output_dir : str
            Directory to save figures
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_trajectory_convergence(self, save: bool = True) -> Dict:
        """
        Analyze training convergence and loss trajectory.
        
        Expected: Loss should decrease and stabilize during training.
        Note: Full parameter trajectory not saved in results for efficiency.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Training loss convergence
        for result in self.results:
            losses = result['training_history']['train_losses']
            T = len(losses)
            axes[0, 0].plot(losses, alpha=0.6, linewidth=1.5)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Validation loss
        for result in self.results:
            losses = result['training_history']['val_losses']
            axes[0, 1].plot(losses, alpha=0.6, linewidth=1.5)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_title('Validation Loss Convergence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training accuracy
        for result in self.results:
            accs = result['training_history']['train_accs']
            axes[1, 0].plot(accs, alpha=0.6, linewidth=1.5)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Training Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Validation accuracy
        for result in self.results:
            accs = result['training_history']['val_accs']
            axes[1, 1].plot(accs, alpha=0.6, linewidth=1.5)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'training_convergence.png', bbox_inches='tight')
            print(f"Saved: training_convergence.png")
        
        # Compute statistics
        final_train_losses = []
        final_val_losses = []
        for result in self.results:
            train_losses = result['training_history']['train_losses']
            val_losses = result['training_history']['val_losses']
            # Check convergence: is loss stable in last 10 epochs?
            if len(train_losses) >= 10:
                last_10_std = np.std(train_losses[-10:])
                final_train_losses.append(last_10_std)
                final_val_losses.append(np.std(val_losses[-10:]))
        
        analysis = {
            'mean_final_train_loss_std': np.mean(final_train_losses) if final_train_losses else 0,
            'mean_final_val_loss_std': np.mean(final_val_losses) if final_val_losses else 0,
            'convergence_observed': np.mean(final_train_losses) < 0.01 if final_train_losses else True
        }
        
        return analysis
    
    def analyze_hessian_spectrum(self, save: bool = True) -> Dict:
        """
        Analyze Hessian eigenspectrum and curvature distribution.
        
        Expected: Large eigenvalues indicate steep directions, small eigenvalues indicate flat directions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Note: For diagonal Hessian, eigenvalues are the diagonal elements
        all_eigenvalues = []
        
        for result in self.results:
            # If diagonal Hessian was used, the eigenvalues are stored directly
            # Otherwise we'd need to compute them
            if 'hessian_eigenvalues' in result:
                eigenvalues = result['hessian_eigenvalues']
            else:
                # Approximate from KL decomposition or other available data
                print("Note: Full eigenspectrum not available (diagonal approximation used)")
                eigenvalues = None
            
            if eigenvalues is not None:
                all_eigenvalues.append(eigenvalues)
        
        if all_eigenvalues:
            # 1. Eigenvalue spectrum
            for eigs in all_eigenvalues:
                sorted_eigs = np.sort(eigs)[::-1]
                axes[0].plot(range(len(sorted_eigs)), sorted_eigs, alpha=0.5)
            
            axes[0].set_xlabel('Eigenvalue Index (sorted)')
            axes[0].set_ylabel('Eigenvalue Magnitude')
            axes[0].set_title('Hessian Eigenspectrum')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Cumulative explained variance
            for eigs in all_eigenvalues:
                sorted_eigs = np.sort(eigs)[::-1]
                cumsum = np.cumsum(sorted_eigs) / np.sum(sorted_eigs)
                axes[1].plot(range(len(cumsum)), cumsum, alpha=0.5)
            
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('Cumulative Proportion')
            axes[1].set_title('Cumulative Explained Curvature')
            axes[1].axhline(y=0.9, color='r', linestyle='--', label='90%')
            axes[1].axhline(y=0.95, color='g', linestyle='--', label='95%')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Eigenspectrum analysis\nrequires full Hessian\n(not available with diagonal approx.)',
                        ha='center', va='center', fontsize=12)
            axes[1].text(0.5, 0.5, 'Use use_diagonal_hessian=False\nfor full spectrum analysis',
                        ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'hessian_spectrum.png', bbox_inches='tight')
            print(f"Saved: hessian_spectrum.png")
        
        return {'eigenspectrum_available': len(all_eigenvalues) > 0}
    
    def analyze_kl_components_detailed(self, save: bool = True) -> Dict:
        """
        Detailed breakdown of KL divergence components.
        
        Expected: 
        - A_r dominates when posterior aligns with top prior directions
        - R_r shows contribution from residual directions
        - FD_T reflects curvature sensitivity
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract KL components across seeds
        kl_data = []
        for i, result in enumerate(self.results):
            decomp = result['kl_decomposition']
            kl_data.append({
                'seed': result['seed'],
                'total_kl': decomp.get('total_kl', 0),
                **decomp
            })
        
        df = pd.DataFrame(kl_data)
        
        # 1. Component magnitudes
        if 'mean_alignment' in df.columns:
            # Diagonal approximation
            components = ['mean_alignment', 'trace_term', 'log_det_term', 'dimension_term']
            labels = ['Mean\nAlignment', 'Trace\nTerm', 'Log-Det\nTerm', 'Dimension\nTerm']
        else:
            # Full matrix
            components = ['top_subspace_alignment', 'complement_alignment', 
                         'trace_top_subspace', 'trace_residual', 'log_det_term']
            labels = ['Top\nSubspace\nAlign.', 'Complement\nAlign.', 
                     'Trace\nTop', 'Trace\nResidual', 'Log-Det']
        
        means = [df[comp].mean() for comp in components if comp in df.columns]
        stds = [df[comp].std() for comp in components if comp in df.columns]
        valid_labels = [labels[i] for i, comp in enumerate(components) if comp in df.columns]
        
        axes[0, 0].bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xticks(range(len(means)))
        axes[0, 0].set_xticklabels(valid_labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Contribution')
        axes[0, 0].set_title('KL Decomposition: Component Magnitudes')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Relative contributions (percentage)
        abs_means = [abs(m) for m in means]
        total = sum(abs_means)
        if total > 0:
            percentages = [100 * m / total for m in abs_means]
            axes[0, 1].pie(percentages, labels=valid_labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Relative Contribution to |KL|')
        
        # 3. Total KL vs components
        axes[1, 0].scatter(df['total_kl'], df[components[0]], alpha=0.6, label=valid_labels[0])
        if len(components) > 1:
            axes[1, 0].scatter(df['total_kl'], df[components[1]], alpha=0.6, label=valid_labels[1])
        axes[1, 0].set_xlabel('Total KL Divergence')
        axes[1, 0].set_ylabel('Component Value')
        axes[1, 0].set_title('Component vs Total KL')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Variance across seeds
        component_vars = [df[comp].var() for comp in components if comp in df.columns]
        axes[1, 1].bar(range(len(component_vars)), component_vars, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xticks(range(len(component_vars)))
        axes[1, 1].set_xticklabels(valid_labels, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Variance Across Seeds')
        axes[1, 1].set_title('Component Stability')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'kl_components_detailed.png', bbox_inches='tight')
            print(f"Saved: kl_components_detailed.png")
        
        # Analysis summary
        analysis = {
            'dominant_component': valid_labels[np.argmax([abs(m) for m in means])],
            'dominant_percentage': max(percentages) if total > 0 else 0,
            'component_stability': {label: df[comp].std() / abs(df[comp].mean()) 
                                   for label, comp in zip(valid_labels, components) 
                                   if comp in df.columns and abs(df[comp].mean()) > 1e-6}
        }
        
        return analysis
    
    def analyze_bound_tightness(self, save: bool = True) -> Dict:
        """
        Analyze bound tightness: gap between bound and actual risk.
        
        Expected: Tighter bounds with better hyperparameters, longer trajectories.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        test_risks = [r['test_risk'] for r in self.results]
        traj_bounds = [r['trajectory_bound']['pac_bayes_bound'] for r in self.results]
        base_bounds = [r['baseline_bound']['pac_bayes_bound'] for r in self.results]
        train_risks = [r['train_risk'] for r in self.results]
        kl_divs = [r['trajectory_bound']['kl_divergence'] for r in self.results]
        
        # 1. Bound vs actual risk
        x = np.arange(len(test_risks))
        width = 0.25
        
        axes[0, 0].bar(x - width, train_risks, width, label='Train Risk', alpha=0.7)
        axes[0, 0].bar(x, test_risks, width, label='Test Risk', alpha=0.7)
        axes[0, 0].bar(x + width, traj_bounds, width, label='Trajectory Bound', alpha=0.7)
        axes[0, 0].bar(x + 2*width, base_bounds, width, label='Baseline Bound', alpha=0.7)
        axes[0, 0].set_xlabel('Experiment (Seed)')
        axes[0, 0].set_ylabel('Risk / Bound')
        axes[0, 0].set_title('Bound Comparison Across Seeds')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Gap analysis
        traj_gaps = [b - t for b, t in zip(traj_bounds, test_risks)]
        base_gaps = [b - t for b, t in zip(base_bounds, test_risks)]
        
        axes[0, 1].violinplot([traj_gaps, base_gaps], positions=[1, 2], 
                             showmeans=True, showextrema=True)
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['Trajectory\nBound Gap', 'Baseline\nBound Gap'])
        axes[0, 1].set_ylabel('Bound - Test Risk')
        axes[0, 1].set_title('Bound Gap Distribution')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. KL vs bound tightness
        axes[1, 0].scatter(kl_divs, traj_gaps, alpha=0.6, s=100, label='Trajectory')
        axes[1, 0].set_xlabel('KL Divergence')
        axes[1, 0].set_ylabel('Bound Gap')
        axes[1, 0].set_title('KL Divergence vs Bound Tightness')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Relative improvement
        improvements = [(base - traj) / base * 100 for base, traj in zip(base_gaps, traj_gaps)]
        axes[1, 1].hist(improvements, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(improvements), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(improvements):.1f}%')
        axes[1, 1].set_xlabel('Improvement (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Trajectory Bound Improvement over Baseline')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'bound_tightness_analysis.png', bbox_inches='tight')
            print(f"Saved: bound_tightness_analysis.png")
        
        analysis = {
            'mean_trajectory_gap': np.mean(traj_gaps),
            'mean_baseline_gap': np.mean(base_gaps),
            'mean_improvement_percent': np.mean(improvements),
            'trajectory_tighter': np.mean(traj_gaps) < np.mean(base_gaps),
            'gap_reduction': np.mean(base_gaps) - np.mean(traj_gaps)
        }
        
        return analysis
    
    def create_summary_table(self, save: bool = True) -> pd.DataFrame:
        """
        Create comprehensive summary table with all metrics.
        """
        data = []
        
        for result in self.results:
            row = {
                'Seed': result['seed'],
                'Train Risk': result['train_risk'],
                'Val Risk': result['val_risk'],
                'Test Risk': result['test_risk'],
                'Trajectory Bound': result['trajectory_bound']['pac_bayes_bound'],
                'Baseline Bound': result['baseline_bound']['pac_bayes_bound'],
                'KL Divergence': result['trajectory_bound']['kl_divergence'],
                'Trajectory Length': result['trajectory_length'],
                'Bound Gap (Traj)': result['trajectory_bound']['pac_bayes_bound'] - result['test_risk'],
                'Bound Gap (Base)': result['baseline_bound']['pac_bayes_bound'] - result['test_risk'],
                'Final Train Acc': result['training_history']['train_accs'][-1],
                'Final Val Acc': result['training_history']['val_accs'][-1]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add summary statistics
        summary_row = {
            'Seed': 'Mean ± Std',
            **{col: f"{df[col].mean():.4f} ± {df[col].std():.4f}" 
               for col in df.columns if col != 'Seed'}
        }
        
        # Convert to display format
        df_display = df.copy()
        for col in df_display.columns:
            if col != 'Seed':
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        
        if save:
            # Save detailed table
            df.to_csv(self.output_dir / 'detailed_results.csv', index=False)
            
            # Save LaTeX table
            with open(self.output_dir / 'detailed_results.tex', 'w') as f:
                f.write(df_display.to_latex(index=False, escape=False))
            
            print(f"Saved: detailed_results.csv and detailed_results.tex")
        
        return df
    
    def generate_all_analyses(self):
        """
        Generate all analyses and save results.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        print("\n1. Analyzing training convergence...")
        conv_analysis = self.analyze_trajectory_convergence()
        print(f"   Convergence observed: {conv_analysis['convergence_observed']}")
        print(f"   Final loss stability (std): {conv_analysis['mean_final_train_loss_std']:.6f}")
        
        print("\n2. Analyzing Hessian spectrum...")
        spec_analysis = self.analyze_hessian_spectrum()
        
        print("\n3. Analyzing KL components...")
        kl_analysis = self.analyze_kl_components_detailed()
        print(f"   Dominant component: {kl_analysis['dominant_component']}")
        print(f"   Contributes: {kl_analysis['dominant_percentage']:.1f}%")
        
        print("\n4. Analyzing bound tightness...")
        tight_analysis = self.analyze_bound_tightness()
        print(f"   Trajectory bound gap: {tight_analysis['mean_trajectory_gap']:.4f}")
        print(f"   Baseline bound gap: {tight_analysis['mean_baseline_gap']:.4f}")
        print(f"   Improvement: {tight_analysis['mean_improvement_percent']:.1f}%")
        
        print("\n5. Creating summary table...")
        summary_df = self.create_summary_table()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nAll figures saved to: {self.output_dir}/")
        
        return {
            'convergence': conv_analysis,
            'spectrum': spec_analysis,
            'kl_components': kl_analysis,
            'bound_tightness': tight_analysis,
            'summary_table': summary_df
        }
