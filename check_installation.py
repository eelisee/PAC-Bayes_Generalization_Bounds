#!/usr/bin/env python
"""
Installation check script for PAC-Bayes project.
Run this to verify all dependencies are correctly installed.
"""

import sys
import importlib

def check_module(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (>= 3.8)")
        return True
    else:
        print("✗ Python version should be >= 3.8")
        return False

def main():
    """Main check function."""
    print("="*60)
    print("PAC-Bayes Project Installation Check")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    print("\nChecking required packages:")
    print("-"*60)
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn")
    ]
    
    results = []
    for module, package in packages:
        results.append(check_module(module, package))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_ok = python_ok and all(results)
    
    if all_ok:
        print("✓ All dependencies are installed correctly!")
        print("\nYou can now run experiments:")
        print("  python experiments/run_experiment.py")
    else:
        print("✗ Some dependencies are missing.")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Check CUDA availability
    print("\n" + "="*60)
    print("GPU Check")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("✗ CUDA is not available (CPU only)")
            print("  Note: Experiments will run on CPU (slower)")
    except:
        pass
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
