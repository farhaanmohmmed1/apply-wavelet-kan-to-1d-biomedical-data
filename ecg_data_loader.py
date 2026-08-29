'''
ECG Data Loading Utilities for PhysioNet datasets
Handles preprocessing and batching of 1D biomedical signals
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals with proper preprocessing
    """
    def __init__(self, signals, labels, signal_length=5000, normalize=True, 
                 preprocess_type='standard', augmentation=False):
        """
        Args:
            signals: numpy array of shape [num_samples, variable_length]
            labels: numpy array of shape [num_samples]
            signal_length: Fixed length to pad/truncate signals to
            normalize: Whether to normalize signals
            preprocess_type: 'standard' (z-norm), 'minmax', or 'none'
            augmentation: Apply data augmentation (noise, shifting, scaling)
        """
        self.signals = signals
        self.labels = labels
        self.signal_length = signal_length
        self.normalize = normalize
        self.preprocess_type = preprocess_type
        self.augmentation = augmentation
        
        # Preprocess all signals
        self.processed_signals = self._preprocess_signals()
        
        # Calculate class weights for imbalanced data
        self.class_weights = self._calculate_class_weights()
    
    def _preprocess_signals(self):
        """
        Preprocessing pipeline for ECG signals:
        1. Handle variable length (pad or truncate)
        2. Normalize
        3. Remove baseline drift
        """
        processed = []
        
        for signal_data in self.signals:
            # Step 1: Handle variable length
            if len(signal_data) > self.signal_length:
                # Truncate from center to preserve QRS complex
                start = (len(signal_data) - self.signal_length) // 2
                signal_data = signal_data[start:start + self.signal_length]
            elif len(signal_data) < self.signal_length:
                # Pad with zeros
                pad_width = self.signal_length - len(signal_data)
                signal_data = np.pad(signal_data, (pad_width//2, pad_width - pad_width//2), 
                                    mode='constant', constant_values=0)
            
            # Step 2: Remove baseline drift using high-pass filter
            # Critical for ECG: baseline wander is common noise
            if len(signal_data) > 50:  # Only if signal is long enough
                b, a = signal.butter(4, 0.5, 'high', fs=100)  # High-pass filter at 0.5 Hz
                try:
                    signal_data = signal.filtfilt(b, a, signal_data)
                except:
                    pass  # If filtering fails, keep original
            
            # Step 3: Normalize
            if self.normalize:
                if self.preprocess_type == 'standard':
                    # Z-score normalization (mean=0, std=1)
                    mean = np.mean(signal_data)
                    std = np.std(signal_data)
                    if std > 0:
                        signal_data = (signal_data - mean) / std
                elif self.preprocess_type == 'minmax':
                    # Min-max normalization (0 to 1)
                    min_val = np.min(signal_data)
                    max_val = np.max(signal_data)
                    if max_val - min_val > 0:
                        signal_data = (signal_data - min_val) / (max_val - min_val)
            
            processed.append(signal_data.astype(np.float32))
        
        return np.array(processed)
    
    def _calculate_class_weights(self):
        """
        Calculate weights for imbalanced ECG dataset
        Inverse of class frequency - rare arrhythmias get higher weight
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = torch.zeros(int(unique.max()) + 1)
        
        for cls, count in zip(unique, counts):
            # Weight inversely proportional to frequency
            weights[int(cls)] = total / (count * len(unique))
        
        return weights / weights.sum() * len(unique)  # Normalize
    
    def _augment_signal(self, signal_data):
        """
        Data augmentation for ECG:
        - Additive Gaussian noise
        - Shifting
        - Scaling (within physiological range)
        """
        if not self.augmentation:
            return signal_data
        
        # Random noise (SNR-aware)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, signal_data.shape)
            signal_data = signal_data + noise
        
        # Random vertical shift (baseline wander simulation)
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            signal_data = signal_data + shift
        
        # Random scaling (amplitude variation)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            signal_data = signal_data * scale
        
        return signal_data
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        signal_data = self.processed_signals[idx].copy()
        
        # Apply augmentation during training
        if self.augmentation:
            signal_data = self._augment_signal(signal_data)
        
        return {
            'signal': torch.from_numpy(signal_data).float(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'idx': idx
        }


def create_ecg_dataloaders(X_train, y_train, X_val, y_val, X_test=None, y_test=None,
                          batch_size=32, signal_length=5000, num_workers=0,
                          preprocess_type='standard', augmentation=True):
    """
    Create PyTorch DataLoaders for ECG data
    
    Args:
        X_train, y_train: Training signals and labels
        X_val, y_val: Validation signals and labels
        X_test, y_test: Test signals and labels (optional)
        batch_size: Batch size for training
        signal_length: Fixed ECG signal length
        num_workers: Number of workers for data loading
        preprocess_type: Preprocessing type ('standard', 'minmax', 'none')
        augmentation: Apply augmentation to training data
    
    Returns:
        Dictionary with dataloaders and class weights
    """
    # Create datasets
    train_dataset = ECGDataset(
        X_train, y_train, 
        signal_length=signal_length,
        normalize=True,
        preprocess_type=preprocess_type,
        augmentation=augmentation  # Only for training
    )
    
    val_dataset = ECGDataset(
        X_val, y_val,
        signal_length=signal_length,
        normalize=True,
        preprocess_type=preprocess_type,
        augmentation=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'class_weights': train_dataset.class_weights
    }
    
    # Test loader (optional)
    if X_test is not None and y_test is not None:
        test_dataset = ECGDataset(
            X_test, y_test,
            signal_length=signal_length,
            normalize=True,
            preprocess_type=preprocess_type,
            augmentation=False
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def load_physionet_ecg_example():
    """
    Example function showing how to load PhysioNet ECG data
    This is a template - adjust for your specific dataset
    
    PhysioNet sources:
    - MIT-BIH Arrhythmia Database (https://www.physionet.org/content/mitdb/1.0.0/)
    - China Physiological Signal Challenge (http://2018.icbeb.org/)
    - ECG-ViEW (https://www.physionet.org/content/ecg-view-ii/1.0/)
    """
    print("PhysioNet ECG Loading Template")
    print("-" * 50)
    print("To load data, use:")
    print("1. wfdb library: pip install wfdb")
    print("2. Example code:")
    print("""
    import wfdb
    
    # Load MIT-BIH record
    record = wfdb.rdrecord('mit-bih-arrhythmia-database/100')
    signal = record.p_signal[:, 0]  # First channel
    annotations = wfdb.rdann('mit-bih-arrhythmia-database/100', 'atr')
    
    # Labels: N (normal), A (AFIB), V (PVC), etc.
    """)
    
    # Placeholder return
    return None, None, None, None


if __name__ == "__main__":
    print("ECG Data Loading Module")
    print("=" * 50)
    
    # Test with dummy data
    print("\nCreating dummy ECG dataset for testing...")
    X_train = np.random.randn(100, 4800).astype(np.float32)  # 100 samples of variable length
    y_train = np.random.randint(0, 5, 100)  # 5 classes
    
    X_val = np.random.randn(20, 4800).astype(np.float32)
    y_val = np.random.randint(0, 5, 20)
    
    # Create dataloaders
    dataloaders = create_ecg_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=32,
        signal_length=5000,
        preprocess_type='standard',
        augmentation=True
    )
    
    print(f"Train loader: {len(dataloaders['train'])} batches")
    print(f"Val loader: {len(dataloaders['val'])} batches")
    print(f"Class weights: {dataloaders['class_weights']}")
    
    # Test a batch
    batch = next(iter(dataloaders['train']))
    print(f"\nBatch signal shape: {batch['signal'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print(f"Signal sample (first 10 values): {batch['signal'][0, :10]}")
