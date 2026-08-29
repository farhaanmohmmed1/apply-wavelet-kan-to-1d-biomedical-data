# ECG Arrhythmia Detection with Wav-KAN - Key Insights & Troubleshooting

## Quick Reference: What You Need to Know

### Files Created for You

| File | Purpose |
|------|---------|
| **ecg_kan.py** | Core ECG-KAN model with wavelet implementations |
| **ecg_data_loader.py** | Data loading, preprocessing, augmentation |
| **ecg_train.py** | Training loop, validation, model saving |
| **ecg_quickstart.py** | Run immediately with synthetic data (demo) |
| **ECG_IMPLEMENTATION_GUIDE.md** | Comprehensive 7-step guide |
| **KEY_INSIGHTS.md** | This file - things to know |

### Run Immediately

```bash
# Test the pipeline with synthetic data
python ecg_quickstart.py

# This will:
# 1. Generate synthetic ECG data
# 2. Create Wav-KAN model
# 3. Train for 30 epochs
# 4. Evaluate on test set
# 5. Save results to ./ecg_models/
```

---

## Why Wav-KAN is Good for ECG

### ECG Signal Characteristics

```
Normal ECG has 5 components:
- P-wave: 0.1-0.25 seconds (atrial depolarization) - LOW FREQUENCY
- QRS complex: 0.06-0.12 seconds (ventricular depolarization) - HIGH FREQUENCY  
- T-wave: 0.2-0.4 seconds (ventricular repolarization) - MEDIUM FREQUENCY
- Baseline: Very slow drift (0.05-1 Hz) - NOISE
- Noise: EMG, 60 Hz interference - NOISE
```

### How Wavelets Help

```
Traditional CNN/RNN:
- CNNs: Designed for 2D images, awkward for 1D sequences
- RNNs: Slow, hard to interpret, hard to debug

Wav-KAN:
- 1D signals: Natural input (no reshaping to 2D)
- Multi-resolution: Captures P-wave AND QRS complex
- Interpretable: Can see which frequencies matter
- Noise-robust: Wavelets separate signal from noise
- Efficient: Works with smaller datasets
```

**Mathematical intuition:**
```
Wavelet = "Zoom lens" for signals
- Zoom in on high-frequency (QRS) → sharp wavelets
- Zoom out on low-frequency (baseline) → smooth wavelets
- Noise gets absorbed by filtering

Morlet wavelet = oscillating Gaussian
  Good for: Detecting oscillations in QRS
  
Mexican Hat = negative Laplacian of Gaussian  
  Good for: Detecting edges (wave boundaries)
```

---

## Critical Preprocessing Steps

### Why Preprocessing Matters

```python
Raw ECG signal → Has:
1. Baseline wander (0.05-1 Hz) - electrode drift
2. High-frequency noise (60 Hz EMG)
3. Amplitude variations (different patients, electrodes)
4. Variable lengths (different recording durations)

After preprocessing:
- Stable mean and variance
- Noise removed but signal preserved
- Ready for neural network
```

### Filtering (MOST IMPORTANT)

```python
from scipy import signal

# High-pass filter: Remove slow baseline drift
b, a = signal.butter(4, 0.5, 'high', fs=100)  # Cutoff at 0.5 Hz
filtered = signal.filtfilt(b, a, ecg_signal)

# What this does:
# - Removes baseline wander
# - Preserves P, QRS, T waves
# - Critical for training stability

# If you skip this:
# - Model learns baseline patterns instead of arrhythmias
# - Poor generalization to different patients
# - Training may become unstable
```

### Normalization

```python
# Option 1: Z-score (RECOMMENDED)
mean = ecg.mean()
std = ecg.std()
normalized = (ecg - mean) / std
# Result: mean=0, std=1

# Why: 
# - Neural networks learn better with centered data
# - Handles different electrode amplitudes
# - Standard in ML

# Option 2: Min-max
min_val = ecg.min()
max_val = ecg.max()
normalized = (ecg - min_val) / (max_val - min_val)
# Result: range [0, 1]

# Option 3: Per-signal vs Global
# Per-signal (recommended): normalize each ECG independently
# Global: normalize across entire dataset (NOT recommended for ECG)
```

---

## Handling Class Imbalance (CRITICAL!)

### The Problem

```
Arrhythmia occurrence in real data:
┌─────────────────────────────────────────┐
│ Normal Sinus Rhythm: ████████████ 90%   │
│ Atrial Fibrillation: ██ 5%               │
│ Premature Ventricular Contraction: █ 3% │
│ LBBB: 1%                                 │
│ RBBB: 1%                                 │
└─────────────────────────────────────────┘

Without handling imbalance:
→ Model learns: "Always predict Normal"
→ 90% accuracy but useless for detecting arrhythmias!
```

### Solution 1: Weighted Loss (Used in Our Code)

```python
# Calculate class weights (inverse of frequency)
from collections import Counter

counts = Counter(y_train)
total = len(y_train)
class_weights = {}
for cls, count in counts.items():
    class_weights[cls] = total / (count * len(counts))

# Normalize so sum = num_classes
weights = np.array(list(class_weights.values()))
weights = weights / weights.sum() * len(weights)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))

# Effect:
# Normal (90% of data) → weight = 0.56
# AFIB (5% of data) → weight = 10.0  ← 18x higher!
# → Rare arrhythmias penalize model more when misclassified
```

### Solution 2: Oversampling (Alternative)

```python
from imblearn.over_sampling import SMOTE

# Synthetic Minority Over-sampling Technique
X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Creates synthetic examples of rare classes
# New distribution: more balanced
# Use with training data only!
```

### Solution 3: Focal Loss (Advanced)

```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focuses training on hard-to-classify examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Use in training
criterion = FocalLoss()
```

---

## Medical Evaluation Metrics

### Why Accuracy is NOT Enough

```
Example: Arrhythmia detector with 95% accuracy
  - But what if it always predicts "Normal"?
  - 90% of data is Normal → 90% accuracy
  - But 0% of arrhythmias detected → USELESS!

Medical requirement: HIGH sensitivity (catch most arrhythmias)
  Accept some false alarms (low specificity)
  → Better to have 1 false alarm than 1 missed arrhythmia
```

### Key Metrics

```python
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# For arrhythmia A:
# True Positives (TP): Correctly detected arrhythmia A
# False Positives (FP): Incorrectly predicted arrhythmia A  
# False Negatives (FN): Missed arrhythmia A
# True Negatives (TN): Correctly identified as not A

# Sensitivity (Recall, TPR)
sensitivity = TP / (TP + FN)
# "Of actual arrhythmias, how many did we catch?"
# Goal: HIGH (minimize missed arrhythmias)
# Medical priority: HIGHEST

# Specificity (TNR)
specificity = TN / (TN + FP)
# "Of normal cases, how many identified correctly?"
# Goal: HIGH (minimize false alarms)
# Medical priority: MEDIUM

# Precision
precision = TP / (TP + FP)
# "Of predicted arrhythmias, how many are correct?"

# F1-Score (Harmonic mean)
f1 = 2 * (precision * recall) / (precision + recall)
# Good for imbalanced data

# ROC-AUC
auc = roc_auc_score(y_true, y_proba)
# Measures performance at all thresholds
# Perfect = 1.0, Random = 0.5
```

### Example Evaluation

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, 
                          target_names=['Normal', 'AFIB', 'PVC', 'LBBB', 'RBBB']))

# Output:
#              precision    recall  f1-score   support
#       Normal       0.92      0.95      0.93        90
#        AFIB       0.85      0.80      0.82         5
#         PVC       0.88      0.75      0.81         3
#        LBBB       0.90      0.50      0.64         1
#        RBBB       0.75      0.50      0.60         1
#
#    accuracy                           0.91       100
#   macro avg       0.86      0.70      0.76       100
# weighted avg      0.91      0.91      0.91       100

# Read this as:
# - Normal: 95% sensitivity (good)
# - AFIB: 80% sensitivity (acceptable, but could improve)
# - PVC: 75% sensitivity (room for improvement)
# - LBBB: 50% sensitivity (very low - needs more data)
```

---

## Common Issues & Solutions

### Issue 1: Model Predicts Only One Class

```python
# Symptom: Accuracy = 90%, but only predicting "Normal"

# Cause: Class imbalance without weighted loss

# Solution:
# 1. Make sure using class_weights
trainer.train(..., class_weights=dataloaders['class_weights'])

# 2. Check weights are calculated
print(dataloaders['class_weights'])
# Should see higher weights for rare classes

# 3. Verify class distribution
print(np.bincount(y_train))
# If ratio > 10:1, may need additional techniques
```

### Issue 2: Validation Loss Not Decreasing

```python
# Symptom: Training loss goes down but validation loss stays flat

# Cause 1: Data preprocessing not working
# Solution:
from ecg_data_loader import ECGDataset
dataset = ECGDataset(X_train, y_train)
sample_signal = dataset.processed_signals[0]
print(f"Mean: {sample_signal.mean()}, Std: {sample_signal.std()}")
# Should be close to 0 and 1 respectively

# Cause 2: Learning rate too high
# Solution:
trainer.train(..., lr=5e-4)  # Reduce from 1e-3

# Cause 3: Model too simple for data
# Solution:
ECG_KAN(5000, [512, 256, 128], 5)  # Increase capacity

# Cause 4: Insufficient data
# Solution:
# - Use data augmentation (already in code)
# - Collect more ECG signals
# - Use transfer learning
```

### Issue 3: CUDA Out of Memory

```python
# Symptom: RuntimeError: CUDA out of memory

# Solutions (in order):
# 1. Reduce batch size
batch_size=16  # instead of 32

# 2. Reduce hidden layer size
ECG_KAN(5000, [128, 64], 5)  # instead of [256, 128, 64]

# 3. Reduce signal length
signal_length=2500  # instead of 5000

# 4. Use gradient checkpointing (advanced)
# or switch to CPU
device = 'cpu'  # will be slow but works

# 5. Check what's using memory
torch.cuda.memory_summary()
```

### Issue 4: Training is Slow

```python
# Symptoms: Takes forever to train

# Cause 1: Using CPU instead of GPU
# Solution:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cause 2: wavelet_type='morlet' (more complex)
# Solution:
model = ECG_KAN(..., wavelet_type='mexican_hat')  # Faster

# Cause 3: Signal too long, model too big
# Solution:
signal_length=2500
ECG_KAN(2500, [128, 64], 5)  # Smaller model

# Cause 4: Too many data loader workers
# Solution:
num_workers=0  # for Windows
# or num_workers=2 for Linux/Mac

# 5. Use mixed precision training (advanced)
from torch.cuda.amp import autocast
with autocast():
    outputs = model(signals)
```

### Issue 5: Model Overfitting

```python
# Symptom: Training loss very low, validation loss high

# Cause 1: Insufficient regularization
# Solution:
model = ECG_KAN(..., dropout_rate=0.4)  # Increase from 0.2

# Cause 2: Dataset too small
# Solution:
# - Increase augmentation
# - Use K-fold cross-validation
# - Transfer learning from large ECG dataset
# - Collect more data

# Cause 3: Model too large
# Solution:
ECG_KAN(5000, [64, 32], 5)  # Simpler model

# Cause 4: No early stopping
# Solution:
trainer.train(..., early_stopping_patience=10)  # Make sure this is on
```

---

## Dataset Recommendations

### Best ECG Datasets

1. **MIT-BIH Arrhythmia Database** (Most common)
   - 48 records, 30 min each
   - 5 classes: N, V, A, L, R
   - 100 Hz sampling
   - Download: `pip install wfdb`
   ```python
   import wfdb
   record = wfdb.rdrecord('mit-bih-arrhythmia-database/100')
   ```

2. **CPSC2018** (China Physiological Signal Challenge)
   - 10,000 12-lead ECGs
   - Well-balanced classes
   - 500 Hz sampling

3. **ECG-ViEW**
   - PhysioNet dataset
   - Various arrhythmias

### Minimum Data Requirements

```
For 5-class arrhythmia detection:
Minimum: 500 samples total
  - 350 training
  - 75 validation
  - 75 test

Recommended: 2000+ samples
  - Better generalization
  - More robust evaluation

With data augmentation: Can work with 300 samples
```

---

## Parameter Tuning Guide

### Learning Rate

```python
# Too high (e.g., 1.0): Loss jumps around wildly
# Too low (e.g., 1e-6): Barely learns, very slow
# Good range: 1e-4 to 1e-2

# For ECG: 1e-3 usually works well
# If training unstable: reduce to 5e-4
# If learning too slow: increase to 5e-3
```

### Batch Size

```python
# Small batch (8): Noisy gradients, unstable, overfitting
# Large batch (256): Smooth gradients, fast, underfitting
# Sweet spot: 32-64

# For ECG signals: 32 is usually good
# If GPU memory limited: 16
# If want faster convergence: 64
```

### Dropout Rate

```python
# 0: No regularization, prone to overfitting
# 0.5: Strong regularization, underfitting
# Good range: 0.2-0.4

# For ECG: 0.2 (20%) is default
# If overfitting: increase to 0.4
# If underfitting: decrease to 0.1
```

### Hidden Layer Sizes

```python
# Too small [32, 16]: Underfitting
# Too large [512, 256, 128]: Overfitting, slow
# Good range: [128-256, 64-128, 32-64]

# For ECG: [256, 128, 64] is good starting point
# Smaller dataset: [128, 64]
# Larger dataset: [512, 256, 128]
```

### Wavelet Type

```python
# Morlet: Best for ECG, captures frequency
#   - Pros: Smooth, good interpretability
#   - Cons: Slower computation
#   - Use when: accuracy is priority

# Mexican Hat: Good for edge detection
#   - Pros: Fast, simple
#   - Cons: Less frequency info
#   - Use when: speed is priority, dataset large

# DOG: Between morlet and mexican hat
# Meyer: Very smooth, slower
# Shannon: For very high-frequency content
```

---

## Deployment Checklist

Before deploying to production:

- [ ] Tested on real ECG data (not just synthetic)
- [ ] Validated sensitivity > 90% for arrhythmias
- [ ] Validated specificity > 85% for normal cases
- [ ] Evaluated on held-out test set
- [ ] Tested on different ECG leads
- [ ] Tested on different patient populations
- [ ] Evaluated on different equipment/amplitudes
- [ ] Model size acceptable for deployment (< 50 MB)
- [ ] Inference time acceptable (< 100 ms per signal)
- [ ] Saved model checkpoint and results
- [ ] Documented any preprocessing steps
- [ ] Created inference script
- [ ] Set up monitoring for model drift

---

## Next Steps

1. **Get Real Data**: Download MIT-BIH Arrhythmia Database
   ```bash
   pip install wfdb
   python # and load one record to familiarize yourself
   ```

2. **Modify ecg_train.py**: Replace synthetic data with real data

3. **Experiment with Wavelets**: Compare Morlet vs Mexican Hat

4. **Tune Hyperparameters**: Try different lr, batch_size, architecture

5. **Validate Properly**: Use K-fold cross-validation

6. **Deploy**: Save model and create inference script

---

## Resources

- **Paper**: Wav-KAN https://arxiv.org/abs/2405.12832
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **ECG Interpretation**: https://en.wikipedia.org/wiki/Electrocardiography
- **PhysioNet**: https://www.physionet.org/
- **Wavelet Theory**: https://en.wikipedia.org/wiki/Wavelet
- **WFDB Python**: https://wfdb.readthedocs.io/

---

Good luck with your ECG arrhythmia detection! 🫀🚀

