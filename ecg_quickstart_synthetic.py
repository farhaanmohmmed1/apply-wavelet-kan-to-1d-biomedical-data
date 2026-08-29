'''
Quick Start: ECG Arrhythmia Detection with Wav-KAN
Using SYNTHETIC data (for testing/demo without internet)
Run: python ecg_quickstart_synthetic.py
'''

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Import our modules
from ecg_kan import ECG_KAN
from ecg_data_loader import create_ecg_dataloaders
from ecg_train import ECGTrainer


def generate_synthetic_ecg_data(num_samples=200, signal_length=5000):
    """
    Generate synthetic ECG data for demonstration
    Better than before: longer signals, more realistic patterns
    """
    print("Generating synthetic ECG data for demonstration...")
    
    def create_synthetic_beat():
        """Create one synthetic heartbeat (~280 ms @ 360 Hz = ~100 samples)"""
        t = np.linspace(0, 2*np.pi, 100)
        # P-wave (0.1 sec)
        p_wave = 0.3 * np.exp(-((t - np.pi/4) ** 2) / 0.2) * np.sin(t)
        # QRS complex (0.08 sec) - main feature
        qrs = -1.0 * np.exp(-((t - np.pi) ** 2) / 0.05) * np.sin(2*t)
        # T-wave (0.2 sec)
        t_wave = 0.5 * np.exp(-((t - 5*np.pi/4) ** 2) / 0.3) * np.sin(t)
        return p_wave + qrs + t_wave
    
    X = []
    y = []
    
    print(f"Generating {num_samples} synthetic ECG signals...")
    for i in range(num_samples):
        # Generate heartbeats to fill signal_length
        signal = []
        num_beats = signal_length // 100 + 1
        
        for beat_idx in range(num_beats):
            beat_pattern = create_synthetic_beat()
            class_label = i % 5  # Cycle through 5 classes
            
            if class_label == 0:  # Normal
                # Normal baseline, clean
                variation = np.random.normal(0, 0.05, 100)
                beat_pattern = beat_pattern * 1.0
                
            elif class_label == 1:  # AFIB - irregular rhythm
                # Irregular intervals, variable amplitude
                variation = np.random.normal(0, 0.15, 100)
                beat_pattern = beat_pattern * np.random.uniform(0.7, 1.3)
                # Add jitter to beat timing (irregular)
                if beat_idx % 3 == 0:
                    beat_pattern = beat_pattern * 0.85  # Skip a beat
                
            elif class_label == 2:  # PVC - extra beat
                variation = np.random.normal(0, 0.08, 100)
                # Every 5th beat is abnormally large (PVC)
                if beat_idx % 5 == 0:
                    beat_pattern = beat_pattern * 1.6  # Premature, larger
                else:
                    beat_pattern = beat_pattern * 1.0
                
            elif class_label == 3:  # LBBB - widened QRS (Left Bundle Branch Block)
                variation = np.random.normal(0, 0.07, 100)
                # Wider, slower QRS complex
                beat_pattern = beat_pattern * 1.3
                # Add low-frequency component (slower depolarization)
                beat_pattern = beat_pattern + 0.2 * np.sin(2 * t)
                
            else:  # class_label == 4: RBBB - Right Bundle Branch Block
                variation = np.random.normal(0, 0.07, 100)
                # Similar to LBBB but different pattern
                beat_pattern = beat_pattern * 1.2
                beat_pattern = beat_pattern + 0.15 * np.cos(2 * t)
            
            signal.extend(beat_pattern + variation)
        
        # Truncate or pad to exact length
        signal = signal[:signal_length]
        if len(signal) < signal_length:
            signal = np.pad(signal, (0, signal_length - len(signal)), mode='constant')
        
        X.append(signal)
        y.append(class_label)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_samples} signals...")
    
    print(f"✓ Generated {len(X)} synthetic ECG signals")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    """
    Complete ECG Arrhythmia Detection Pipeline with Synthetic Data
    """
    print("=" * 70)
    print("ECG ARRHYTHMIA DETECTION WITH WAV-KAN - SYNTHETIC DATA DEMO")
    print("=" * 70)
    print("\n⚠️  Using SYNTHETIC data for demonstration")
    print("   For real results, use: python ecg_quickstart.py (requires internet)")
    
    # Configuration
    config = {
        'signal_length': 5000,
        'num_classes': 5,
        'hidden_sizes': [256, 128, 64],
        'wavelet_type': 'morlet',
        'batch_size': 16,
        'epochs': 50,  # Reasonable for synthetic data
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n📊 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ========== STEP 1: Generate Synthetic Data ==========
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("=" * 70)
    
    X_all, y_all = generate_synthetic_ecg_data(num_samples=200, signal_length=5000)
    
    print(f"✓ Generated {len(X_all)} synthetic signals")
    print(f"  Shape: {X_all.shape}")
    print(f"  Class distribution: {np.bincount(y_all)}")
    
    # ========== STEP 2: Split Data ==========
    print("\n" + "=" * 70)
    print("STEP 2: SPLITTING DATA")
    print("=" * 70)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Data split:")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    print(f"  Class distribution (test):  {np.bincount(y_test)}")
    
    # ========== STEP 3: Create DataLoaders ==========
    print("\n" + "=" * 70)
    print("STEP 3: CREATING DATA LOADERS")
    print("=" * 70)
    
    print("\nCreating PyTorch DataLoaders...")
    dataloaders = create_ecg_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=config['batch_size'],
        signal_length=config['signal_length'],
        num_workers=0,
        preprocess_type='standard',
        augmentation=True
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Test batches: {len(dataloaders['test'])}")
    print(f"  Class weights: {dataloaders['class_weights'].numpy()}")
    
    # ========== STEP 4: Create Model ==========
    print("\n" + "=" * 70)
    print("STEP 4: CREATING WAV-KAN MODEL")
    print("=" * 70)
    
    print(f"\nInitializing ECG_KAN model...")
    print(f"  Architecture: {config['signal_length']} → {config['hidden_sizes']} → {config['num_classes']}")
    print(f"  Wavelet: {config['wavelet_type']}")
    
    model = ECG_KAN(
        input_size=config['signal_length'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes'],
        wavelet_type=config['wavelet_type'],
        dropout_rate=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {config['device']}")
    
    # ========== STEP 5: Train Model ==========
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING MODEL")
    print("=" * 70)
    
    trainer = ECGTrainer(
        model=model,
        device=config['device'],
        model_name='ecg_wavkan_synthetic'
    )
    
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        class_weights=dataloaders['class_weights'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # ========== STEP 6: Save Results ==========
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)
    
    trainer.save_training_results()
    
    # ========== STEP 7: Evaluate on Test Set ==========
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATING ON TEST SET")
    print("=" * 70)
    
    print("\nEvaluating on test set...")
    trainer.model.eval()
    
    test_loss_total = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            signals = batch['signal'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            
            outputs = trainer.model(signals)
            loss = criterion(outputs, labels)
            
            test_loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss_total / len(dataloaders['test'])
    test_acc = 100 * test_correct / test_total
    
    print(f"\n✓ Test Set Performance:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    # Detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\n✓ Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['Normal', 'AFIB', 'PVC', 'LBBB', 'RBBB'],
        digits=4
    ))
    
    print(f"\n✓ Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("✅ SYNTHETIC DATA TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📝 Note: This is SYNTHETIC data (not real ECG)")
    print(f"   - Good for: Testing pipeline, hyperparameter tuning, development")
    print(f"   - Not good for: Real-world deployment, medical validation")
    print(f"\n📊 Results saved to: ./ecg_models/")
    print(f"   - ecg_wavkan_synthetic_best.pt")
    print(f"   - ecg_wavkan_synthetic_results.csv")
    print(f"   - ecg_wavkan_synthetic_curves.png")
    print(f"\n📈 Results:")
    print(f"   - Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch+1})")
    print(f"   - Test accuracy: {test_acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("""
1. 👉 For REAL data, run:
   python ecg_quickstart.py
   (Requires: pip install wfdb)

2. 🔬 Compare synthetic vs real:
   - Run this: python ecg_quickstart_synthetic.py
   - Run real: python ecg_quickstart.py
   - Real data should show more realistic/lower accuracy

3. 🔧 Experiment with hyperparameters:
   - Edit config dict at top of this file
   - Try different learning rates, batch sizes, architectures

4. 📊 Try other wavelet types:
   wavelet_type='mexican_hat'  # Faster
   wavelet_type='dog'          # Different characteristics

5. 🚀 When ready for production:
   - Load REAL MIT-BIH data
   - Use K-fold cross-validation
   - Perform extensive evaluation
   - Validate on clinical datasets
    """)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
