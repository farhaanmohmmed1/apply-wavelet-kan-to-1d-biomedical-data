# Wav-KAN for ECG Arrhythmia Detection - Complete Step-by-Step Guide

## Table of Contents
1. [Overview](#overview)
2. [Key Differences from MNIST](#key-differences)
3. [Step-by-Step Implementation](#step-by-step)
4. [Important Things to Know](#important-things)
5. [Running the Code](#running-the-code)
6. [Troubleshooting](#troubleshooting)

---

## Overview

Wav-KAN is **perfectly suited** for ECG arrhythmia detection because:
- **Wavelets naturally match ECG structure**: ECG signals contain multiple frequency components (P-wave, QRS complex, T-wave, baseline drift)
- **Interpretability**: Unlike black-box neural networks, wavelet-based networks show which signal components matter
- **Noise robustness**: Wavelets can separate noise from signal by analyzing different frequency bands
- **Efficiency**: Works well with limited training data (common in medical AI)

---

## Key Differences from MNIST

| Aspect | MNIST (Original) | ECG (Adapted) |
|--------|------------------|---------------|
| **Input Format** | 2D images (28×28 = 784) | 1D signals (5000+ samples) |
| **Sampling** | None (images) | 100-500 Hz (time series) |
| **Classes** | 10 digits (balanced) | 5-15 arrhythmias (imbalanced) |
| **Noise** | None | High (muscle artifacts, baseline drift) |
| **Preprocessing** | Normalization only | Normalization + filtering + baseline removal |
| **Loss Function** | Standard CrossEntropyLoss | Weighted CrossEntropyLoss (handle imbalance) |
| **Metrics** | Accuracy only | Accuracy + Sensitivity + Specificity + F1 |

### Imbalance Problem Example
```
Normal sinus rhythm:    90% of data
Atrial fibrillation:     5% of data
PVC:                     3% of data
LBBB:                    1% of data
RBBB:                    1% of data
```
→ Model learns to always predict "Normal" → 90% accuracy but useless for arrhythmia detection!  
→ **Solution**: Use weighted loss function (rare arrhythmias get higher weight)

---

## Step-by-Step Implementation

### STEP 1: Prepare Your Data

#### 1a. Download PhysioNet ECG Data

```python
# Option 1: MIT-BIH Arrhythmia Database (most common)
# Install: pip install wfdb
import wfdb

# Download and read a record
record = wfdb.rdrecord('mit-bih-arrhythmia-database/100')
signal = record.p_signal[:, 0]  # First channel (ECG lead II)
sampling_rate = record.fs  # Usually 360 Hz

# Read annotations (arrhythmia labels)
annotations = wfdb.rdann('mit-bih-arrhythmia-database/100', 'atr')
# Symbols: N (normal), A (AFIB), V (PVC), L (LBBB), R (RBBB), etc.

print(f"Signal shape: {signal.shape}")  # e.g., (650000,)
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {len(signal) / sampling_rate} seconds")
```

#### 1b. Segment Signals

ECG records are long (often 30 minutes). Need to segment into fixed-length clips:

```python
def segment_ecg(signal, labels, segment_length=5000, sampling_rate=100):
    """
    Segment long ECG into fixed-length windows
    
    Args:
        signal: 1D ECG array
        labels: Label for each sample
        segment_length: Samples per window (5000 samples @ 100Hz = 50 seconds)
        sampling_rate: Hz
    
    Returns:
        segments: [num_segments, segment_length]
        segment_labels: [num_segments]
    """
    segments = []
    segment_labels = []
    
    for i in range(0, len(signal) - segment_length, segment_length // 2):
        segment = signal[i:i + segment_length]
        
        # Get most common label in this segment
        label_window = labels[i:i + segment_length]
        label = np.bincount(label_window).argmax()
        
        segments.append(segment)
        segment_labels.append(label)
    
    return np.array(segments), np.array(segment_labels)
```

#### 1c. Split Data

```python
from sklearn.model_selection import train_test_split

# Segment all ECG records
all_segments = []
all_labels = []
for record_num in range(100, 124):  # MIT-BIH has records 100-109, 111-119, 121-124
    try:
        record = wfdb.rdrecord(f'mit-bih-arrhythmia-database/{record_num}')
        signal = record.p_signal[:, 0]
        annotations = wfdb.rdann(f'mit-bih-arrhythmia-database/{record_num}', 'atr')
        
        segments, labels = segment_ecg(signal, annotations.symbol)
        all_segments.append(segments)
        all_labels.extend(labels)
    except:
        pass

X = np.vstack(all_segments)
y = np.array(all_labels)

# Map labels to integers
label_map = {'N': 0, 'A': 1, 'V': 2, 'L': 3, 'R': 4}
y = np.array([label_map.get(label, 0) for label in y])

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")
```

---

### STEP 2: Understand Preprocessing

Preprocessing is **CRITICAL** for ECG. The original code doesn't do this!

```python
# This is what the ECG data loader does automatically:

# 1. BASELINE REMOVAL (high-pass filter)
# ECG has slow baseline wander (0.05-1 Hz) that's not physiological
from scipy import signal
b, a = signal.butter(4, 0.5, 'high', fs=100)  # High-pass at 0.5 Hz
filtered_ecg = signal.filtfilt(b, a, ecg_signal)

# 2. NORMALIZATION
# Two options:
# a) Z-score: (x - mean) / std → mean=0, std=1
ecg_normalized = (filtered_ecg - filtered_ecg.mean()) / filtered_ecg.std()

# b) Min-max: (x - min) / (max - min) → range [0, 1]
ecg_normalized = (filtered_ecg - filtered_ecg.min()) / (filtered_ecg.max() - filtered_ecg.min())

# 3. DATA AUGMENTATION (for training only!)
# Add noise: simulate measurement variations
noise = np.random.normal(0, 0.01, ecg_normalized.shape)
augmented = ecg_normalized + noise

# Shift: simulate baseline variations
augmented = augmented + np.random.uniform(-0.1, 0.1)

# Scale: simulate amplitude variations
augmented = augmented * np.random.uniform(0.9, 1.1)
```

---

### STEP 3: Create Data Loaders

The code provides `create_ecg_dataloaders()`:

```python
from ecg_data_loader import create_ecg_dataloaders

dataloaders = create_ecg_dataloaders(
    X_train, y_train,      # Training data
    X_val, y_val,          # Validation data
    X_test, y_test,        # Test data
    batch_size=32,         # Larger batch = more stable gradients
    signal_length=5000,    # Fixed length (50 sec @ 100 Hz)
    num_workers=4,         # Parallel data loading
    preprocess_type='standard',  # z-score normalization
    augmentation=True      # Data augmentation only for training
)

# Returns:
# dataloaders['train']: Training loader
# dataloaders['val']: Validation loader
# dataloaders['test']: Test loader
# dataloaders['class_weights']: Weights for imbalanced classes
```

---

### STEP 4: Create Model

```python
from ecg_kan import ECG_KAN

model = ECG_KAN(
    input_size=5000,           # Signal length
    hidden_sizes=[256, 128, 64],  # Architecture: 5000 → 256 → 128 → 64 → 5
    num_classes=5,             # 5 arrhythmia classes
    wavelet_type='morlet',     # Best for ECG (alternative: 'mexican_hat')
    dropout_rate=0.2           # 20% dropout for regularization
)

# Check model
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Example forward pass
dummy_ecg = torch.randn(32, 5000)  # Batch of 32 signals
output = model(dummy_ecg)
print(f"Output shape: {output.shape}")  # [32, 5]
```

---

### STEP 5: Train Model

```python
from ecg_train import ECGTrainer

trainer = ECGTrainer(
    model=model,
    device='cuda',  # Use GPU if available
    model_name='ecg_wavkan_morlet'
)

# Train with weighted loss (handles class imbalance)
history = trainer.train(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    class_weights=dataloaders['class_weights'],  # Important!
    early_stopping_patience=15  # Stop if no improvement for 15 epochs
)

# This will:
# - Train for up to 100 epochs
# - Early stop if validation loss doesn't improve
# - Save best model automatically
# - Print Sensitivity, Specificity, F1 (medical metrics)
```

---

### STEP 6: Evaluate on Test Set

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load best model
best_model = ECG_KAN(5000, [256, 128, 64], 5, 'morlet')
checkpoint = torch.load('./ecg_models/ecg_wavkan_morlet_best.pt')
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.to(device)
best_model.eval()

# Get predictions on test set
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in dataloaders['test']:
        signals = batch['signal'].to(device)
        labels = batch['label']
        
        outputs = best_model(signals)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Classification Report (sensitivity, specificity, F1 per class)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, 
                          target_names=['Normal', 'AFIB', 'PVC', 'LBBB', 'RBBB']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - ECG Arrhythmia Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

---

## Important Things to Know

### 1. **Wavelets for ECG**

Different wavelets work better for different aspects:

| Wavelet | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Morlet** | Best overall for ECG | Captures frequency + time | Higher computational cost |
| **Mexican Hat** | Edge detection (QRS) | Fast, simple | Less frequency resolution |
| **DOG** | P-wave, T-wave | Good oscillation capture | Can be noisy |
| **Meyer** | Smooth analysis | Good frequency separation | Slow |

**Recommendation**: Start with **Morlet**, then try **Mexican Hat** if too slow.

### 2. **Handling Variable-Length Signals**

PhysioNet signals often have different lengths:

```python
# Option 1: Pad with zeros (simpler)
signal_padded = np.pad(signal, (0, max_length - len(signal)), mode='constant')

# Option 2: Truncate from center (preserves QRS)
start = (len(signal) - target_length) // 2
signal_truncated = signal[start:start + target_length]

# Option 3: Interpolation (resampling)
from scipy.interpolate import interp1d
f = interp1d(np.arange(len(signal)), signal, kind='cubic')
signal_resampled = f(np.linspace(0, len(signal)-1, target_length))
```

### 3. **Class Imbalance**

Medical data is always imbalanced. Solutions:

```python
# 1. Weighted loss (used in our code)
class_weights = torch.tensor([1.0, 10.0, 15.0, 25.0, 25.0])  # Higher for rare classes
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2. Oversampling rare classes
from imblearn.over_sampling import SMOTE
X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)

# 3. Focal Loss (emphasizes hard examples)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()
```

### 4. **Metrics for Medical AI**

**Don't just use accuracy!**

```python
# Sensitivity (TPR) = TP / (TP + FN)
# → "Of arrhythmias, how many did we catch?" (High sensitivity = few missed)
sensitivity = recall_score(y_true, y_pred)

# Specificity (TNR) = TN / (TN + FP)
# → "Of normal cases, how many identified correctly?" (High specificity = few false alarms)
specificity = tn / (tn + fp)

# F1 = 2 * (precision * recall) / (precision + recall)
# → Harmonic mean (good for imbalanced data)
f1 = f1_score(y_true, y_pred)

# ROC-AUC (Area Under the Curve)
# → Measures performance across all thresholds
auc = roc_auc_score(y_true, y_proba)
```

### 5. **Batch Normalization in Medical AI**

Batch norm is critical for ECG but can be tricky:

```python
# Current code uses:
self.bn = nn.BatchNorm1d(out_features)

# Problems:
# - Small batch size can make it unstable
# - Different test-time behavior

# Better alternatives:
# 1. Use larger batch size (32+)
# 2. Use GroupNorm for small batches
self.gn = nn.GroupNorm(num_groups=8, num_channels=out_features)

# 3. Use LayerNorm
self.ln = nn.LayerNorm(out_features)
```

### 6. **Avoiding Overfitting**

Medical datasets are usually small. Use:

```python
# 1. Dropout (in code)
self.dropout = nn.Dropout(0.3)

# 2. L2 regularization (weight decay in optimizer)
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)

# 3. Data augmentation (in data loader)
augmentation=True

# 4. Early stopping (in trainer)
early_stopping_patience=15

# 5. K-fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    # Train model on fold
    pass
```

### 7. **Computational Considerations**

ECG signals are long (5000 samples vs 784 for MNIST):

```python
# Memory usage per sample: 5000 * 4 bytes (float32) = 20 KB
# Batch of 32: 640 KB input
# Hidden layer [256]: 256 * 4 = 1 KB per unit
# Total: manageable on most GPUs

# But if too slow:
# Option 1: Use smaller hidden sizes
ECG_KAN(5000, [128, 64, 32], 5)  # Faster, slightly lower accuracy

# Option 2: Use shorter signal windows
# 2500 samples = 25 seconds @ 100 Hz (still captures QRS)

# Option 3: Downsample signal
# Instead of 100 Hz, use 50 Hz
signal_downsampled = signal[::2]  # Take every 2nd sample
```

---

## Running the Code

### Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision scipy scikit-learn pandas numpy matplotlib tqdm wfdb

# 2. Run training script
python ecg_train.py

# 3. Check results
ls ./ecg_models/
# Output:
# ecg_wavkan_morlet_best.pt (model checkpoint)
# ecg_wavkan_morlet_results.csv (metrics)
# ecg_wavkan_morlet_curves.png (training curves)
```

### Custom Training

```python
import torch
from ecg_kan import ECG_KAN
from ecg_data_loader import create_ecg_dataloaders
from ecg_train import ECGTrainer
import numpy as np

# Your data
X_train = np.load('your_train_signals.npy')
y_train = np.load('your_train_labels.npy')
# ... etc

# Create model and train
model = ECG_KAN(5000, [256, 128, 64], 5, 'morlet')
trainer = ECGTrainer(model, device='cuda')
history = trainer.train(...)
```

---

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution**:
```python
# Reduce batch size
batch_size=16  # instead of 32

# Reduce hidden sizes
hidden_sizes=[128, 64]  # instead of [256, 128, 64]

# Reduce signal length
signal_length=2500  # instead of 5000
```

### Problem: "Validation loss not decreasing"
**Solution**:
```python
# 1. Reduce learning rate
lr=5e-4  # instead of 1e-3

# 2. Check data preprocessing
# Make sure signals are normalized!

# 3. Use different wavelet
wavelet_type='mexican_hat'  # simpler than morlet

# 4. Increase training data
# Collect more ECG signals
```

### Problem: "Model predicts only one class"
**Solution**:
```python
# 1. Make sure to use class_weights!
trainer.train(..., class_weights=dataloaders['class_weights'])

# 2. Check label distribution
print(np.bincount(y_train))
# If heavily imbalanced, may need oversampling

# 3. Use focal loss instead of weighted CE
```

### Problem: "Sensitivity/Specificity are unbalanced"
**Solution**:
```python
# Adjust class weights
# Increase weight for false negatives (missed arrhythmias)
class_weights = torch.tensor([1.0, 20.0, 20.0, 20.0, 20.0])

# Or use weighted F1 score for loss
```

---

## Summary

```
1. Load PhysioNet ECG data
   ↓
2. Segment into fixed-length windows (5000 samples)
   ↓
3. Preprocess: normalize, filter, augment
   ↓
4. Create ECG_KAN model (input=5000, hidden=[256,128,64], output=5)
   ↓
5. Train with weighted loss (handle class imbalance)
   ↓
6. Evaluate: Accuracy, Sensitivity, Specificity, F1
   ↓
7. Save best model and results
```

**Key Advantages of Wav-KAN for ECG**:
- ✅ Wavelets naturally match ECG frequency content
- ✅ Interpretable (vs CNNs/RNNs)
- ✅ Efficient with limited data
- ✅ Noise-robust by design
- ✅ Good for edge detection (QRS complexes)

Good luck! 🚀

