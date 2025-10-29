#!/usr/bin/env python3
"""
Run Comprehensive Analysis on Experimental Results
===================================================

This script loads experimental results and generates all comprehensive analyses
to validate theoretical expectations from the paper.

Usage:
    python experiments/run_comprehensive_analysis.py results/results_20251029_225613.json
    python experiments/run_comprehensive_analysis.py results/*.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.comprehensive_analysis import ComprehensiveAnalyzer


def load_results(result_file: Path) -> dict:
    """Load results from JSON file."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive analysis on experimental results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_comprehensive_analysis.py results/results_20251029_225613.json
  python experiments/run_comprehensive_analysis.py results/*.json

This will generate:
  - trajectory_convergence.png: Parameter trajectory analysis
  - hessian_spectrum.png: Eigenvalue distribution (if available)
  - kl_components_detailed.png: Detailed KL decomposition
  - bound_tightness_analysis.png: Bound gap analysis
  - detailed_results.csv/tex: Comprehensive results table
        """
    )
    
    parser.add_argument(
        'result_files',
        type=str,
        nargs='+',
        help='Path(s) to result JSON file(s)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures/)'
    )
    
    args = parser.parse_args()
    
    # Load all results
    all_results = []
    for result_pattern in args.result_files:
        result_files = list(Path('.').glob(result_pattern))
        if not result_files:
            # Try as direct path
            result_path = Path(result_pattern)
            if result_path.exists():
                result_files = [result_path]
            else:
                print(f"Warning: No files found matching '{result_pattern}'")
                continue
        
        for result_file in result_files:
            print(f"\nLoading: {result_file}")
            data = load_results(result_file)
            
            if isinstance(data, list):
                # List of experiments
                all_results.extend(data)
            elif isinstance(data, dict):
                if 'results' in data:
                    # Dict with 'results' key containing list
                    all_results.extend(data['results'])
                else:
                    # Single experiment dict
                    all_results.append(data)
            else:
                print(f"Warning: Unexpected data format in {result_file}")
    
    if not all_results:
        print("\nError: No results loaded!")
        return 1
    
    print(f"\nLoaded {len(all_results)} experiment(s)")
    
    # Run comprehensive analysis
    analyzer = ComprehensiveAnalyzer(all_results, output_dir=args.output_dir)
    analysis_results = analyzer.generate_all_analyses()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nTraining Convergence:")
    print(f"  - Convergence observed: {analysis_results['convergence']['convergence_observed']}")
    print(f"  - Final loss stability (std): {analysis_results['convergence']['mean_final_train_loss_std']:.6f}")
    
    print("\nKL Decomposition:")
    print(f"  - Dominant component: {analysis_results['kl_components']['dominant_component']}")
    print(f"  - Contribution: {analysis_results['kl_components']['dominant_percentage']:.1f}%")
    
    print("\nBound Tightness:")
    print(f"  - Trajectory bound gap: {analysis_results['bound_tightness']['mean_trajectory_gap']:.4f}")
    print(f"  - Baseline bound gap: {analysis_results['bound_tightness']['mean_baseline_gap']:.4f}")
    print(f"  - Improvement: {analysis_results['bound_tightness']['mean_improvement_percent']:.1f}%")
    print(f"  - Trajectory tighter: {analysis_results['bound_tightness']['trajectory_tighter']}")
    
    print("\n" + "="*60)
    print("Summary Table (first 5 rows):")
    print("="*60)
    print(analysis_results['summary_table'].head().to_string())
    
    print(f"\n\nAll figures saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - training_convergence.png")
    print("  - hessian_spectrum.png")
    print("  - kl_components_detailed.png")
    print("  - bound_tightness_analysis.png")
    print("  - detailed_results.csv")
    print("  - detailed_results.tex")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
