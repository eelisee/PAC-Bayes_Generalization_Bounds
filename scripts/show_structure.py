#!/usr/bin/env python3
"""
Generate a visual tree of the project structure.
"""

import os
from pathlib import Path

def generate_tree(directory, prefix="", max_depth=3, current_depth=0, ignore_dirs=None):
    """Generate a tree structure of the directory."""
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.ipynb_checkpoints', 'venv', 'env', 
                      '.vscode', '.idea', 'data', '.DS_Store', '.eggs', 'build', 
                      'dist', '*.egg-info'}
    
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return
    
    for i, entry in enumerate(entries):
        if entry.name in ignore_dirs or entry.name.startswith('.'):
            continue
        
        is_last = i == len(entries) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{entry.name}")
        
        if entry.is_dir():
            extension_prefix = "    " if is_last else "│   "
            generate_tree(entry, prefix + extension_prefix, max_depth, current_depth + 1, ignore_dirs)

if __name__ == "__main__":
    print("PAC-Bayes_Generalization_Bounds/")
    generate_tree(".", prefix="", max_depth=3)
