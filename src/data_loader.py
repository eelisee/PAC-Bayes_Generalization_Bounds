"""
Data loading and preprocessing module.
======================================

This module provides utilities for loading and preprocessing datasets
for binary classification experiments.

Original Implementation:
- Binary classification conversion for MNIST
- Custom train/val/test splitting with reproducibility
- Preprocessing pipeline for PAC-Bayes experiments

Adapted from:
- Standard torchvision MNIST loader
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional
import os


class MNISTBinaryClassification:
    """
    MNIST dataset loader for binary classification.
    
    Converts MNIST to binary classification problem (e.g., digit 0 vs digit 1).
    Includes normalization and train/validation/test splitting.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        class_0: int = 0,
        class_1: int = 1,
        val_split: float = 0.15,
        normalize: bool = True,
        flatten: bool = True,
        seed: int = 42
    ):
        """
        Initialize MNIST binary classification dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store/load MNIST data
        class_0 : int
            First class label (0-9)
        class_1 : int
            Second class label (0-9)
        val_split : float
            Fraction of training data to use for validation
        normalize : bool
            Whether to normalize pixel values to [0,1]
        flatten : bool
            Whether to flatten images to vectors (for logistic regression)
        seed : int
            Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.class_0 = class_0
        self.class_1 = class_1
        self.val_split = val_split
        self.normalize = normalize
        self.flatten = flatten
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load and process data
        self.train_data, self.val_data, self.test_data = self._load_and_process()
        
    def _load_and_process(self) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """Load MNIST and convert to binary classification."""
        
        # Try multiple methods to load MNIST
        train_X_full, train_y_full, test_X_full, test_y_full = None, None, None, None
        
        # Method 1: Try Keras with SSL verification disabled (macOS fix)
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            from tensorflow import keras
            print("Loading MNIST via Keras (SSL verification disabled)...")
            (train_X_full, train_y_full), (test_X_full, test_y_full) = keras.datasets.mnist.load_data()
            print("✓ MNIST loaded successfully")
        except Exception as e:
            print(f"Keras loading failed: {e}")
            
            # Method 2: Try to load from local cache if it exists
            try:
                keras_cache = os.path.expanduser('~/.keras/datasets/mnist.npz')
                if os.path.exists(keras_cache):
                    print(f"Loading from cache: {keras_cache}")
                    with np.load(keras_cache, allow_pickle=True) as f:
                        train_X_full, train_y_full = f['x_train'], f['y_train']
                        test_X_full, test_y_full = f['x_test'], f['y_test']
                    print("✓ MNIST loaded from cache")
                else:
                    raise FileNotFoundError("No cached MNIST found")
            except Exception as e2:
                print(f"Cache loading failed: {e2}")
                
                # Method 3: Try downloading manually with requests
                try:
                    print("Attempting manual download...")
                    mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
                    cache_dir = os.path.expanduser('~/.keras/datasets')
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_file = os.path.join(cache_dir, 'mnist.npz')
                    
                    # Download without SSL verification
                    import urllib.request
                    import ssl
                    context = ssl._create_unverified_context()
                    
                    print(f"Downloading from {mnist_url}...")
                    urllib.request.urlretrieve(mnist_url, cache_file, context=context)
                    
                    # Load from downloaded file
                    with np.load(cache_file, allow_pickle=True) as f:
                        train_X_full, train_y_full = f['x_train'], f['y_train']
                        test_X_full, test_y_full = f['x_test'], f['y_test']
                    print("✓ MNIST downloaded and loaded")
                    
                except Exception as e3:
                    print(f"Manual download failed: {e3}")
                    raise RuntimeError(
                        "Could not load MNIST dataset. Please try one of these solutions:\n"
                        "1. Download manually from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n"
                        f"   and place it in: {os.path.expanduser('~/.keras/datasets/mnist.npz')}\n"
                        "2. Fix SSL certificates on macOS: /Applications/Python\\ 3.*/Install\\ Certificates.command\n"
                        "3. Use a VPN or different network\n"
                    )
        
        if train_X_full is None:
            raise RuntimeError("Failed to load MNIST dataset")
        
        # Filter for binary classification
        train_X, train_y = self._filter_classes_from_arrays(train_X_full, train_y_full)
        test_X, test_y = self._filter_classes_from_arrays(test_X_full, test_y_full)
        
        # Normalize if requested
        if self.normalize:
            train_X = train_X / 255.0
            test_X = test_X / 255.0
        
        # Flatten if requested
        if self.flatten:
            train_X = train_X.reshape(len(train_X), -1)  # (N, 784)
            test_X = test_X.reshape(len(test_X), -1)
        
        # Convert to tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)
        
        # Create train/validation split
        full_train = TensorDataset(train_X, train_y)
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        
        train_dataset, val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        test_dataset = TensorDataset(test_X, test_y)
        
        return train_dataset, val_dataset, test_dataset
    
    def _filter_classes_from_arrays(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter dataset for binary classification from numpy arrays."""
        # Find indices for our two classes
        mask_0 = labels == self.class_0
        mask_1 = labels == self.class_1
        mask = mask_0 | mask_1
        
        # Filter data and labels
        filtered_data = data[mask]
        filtered_labels = labels[mask]
        
        # Convert labels to binary (0 and 1)
        binary_labels = np.where(filtered_labels == self.class_0, 0, 1)
        
        # Add channel dimension if needed (for compatibility)
        if filtered_data.ndim == 3:
            filtered_data = filtered_data[:, np.newaxis, :, :]
        
        return filtered_data, binary_labels
    
    def get_dataloaders(
        self,
        batch_size: int = 128,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for train, validation, and test sets.
        
        Parameters:
        -----------
        batch_size : int
            Batch size for training
        num_workers : int
            Number of worker processes for data loading
            
        Returns:
        --------
        train_loader, val_loader, test_loader : DataLoader objects
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get input dimension and number of samples."""
        sample_x, _ = self.train_data[0]
        input_dim = sample_x.numel()
        n_samples = len(self.train_data)
        return input_dim, n_samples


def get_mnist_binary(
    data_dir: str = "./data",
    class_0: int = 0,
    class_1: int = 1,
    batch_size: int = 128,
    val_split: float = 0.15,
    flatten: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Convenience function to get MNIST binary classification dataloaders.
    
    Returns:
    --------
    train_loader, val_loader, test_loader, input_dim
    """
    dataset = MNISTBinaryClassification(
        data_dir=data_dir,
        class_0=class_0,
        class_1=class_1,
        val_split=val_split,
        flatten=flatten,
        seed=seed
    )
    
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size)
    input_dim, _ = dataset.get_dimensions()
    
    return train_loader, val_loader, test_loader, input_dim
