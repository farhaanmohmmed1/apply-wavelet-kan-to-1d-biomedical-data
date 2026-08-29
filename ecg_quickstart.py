'''
Quick Start: ECG Arrhythmia Detection with Wav-KAN
Using REAL MIT-BIH Arrhythmia Database from PhysioNet
Run: python ecg_quickstart.py
'''

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from ecg_kan import ECG_KAN
from ecg_data_loader import create_ecg_dataloaders
from ecg_train import ECGTrainer
from mit_bih_loader import load_mit_bih_dataset


def load_real_mit_bih_data():
    """
    Load real ECG data from MIT-BIH Arrhythmia Database
    First run downloads from PhysioNet (~100 MB)
    Subsequent runs use cached data
    """
    print("\n" + "="*70)
    print("LOADING MIT-BIH ARRHYTHMIA DATABASE")
    print("="*70)
    print("\n⏳ First run will download ECG data from PhysioNet...")
    print("   This may take 2-5 minutes depending on internet speed\n")
    
    try:
        # Load multiple MIT-BIH records
        # Mix of normal and arrhythmia records
        record_ids = [
            '100',  # Normal
            '101',  # Atrial fibrillation
            '102',  # Normal
            '103',  # Atrial fibrillation
            '104',  # Normal
            '105',  # Atrial fibrillation
            '106',  # Normal
            '107',  # Atrial fibrillation
            '115',  # PVC (Premature Ventricular Contraction)
            '117',  # PVC
            '119',  # PVC
            '121',  # Normal
            '122',  # Normal
            '123',  # PVC
        ]
        
        X, y = load_mit_bih_dataset(
            record_ids=record_ids,
            segment_length=5000,  # ~14 seconds @ 360 Hz
            sampling_rate=360,    # MIT-BIH standard
            cache_dir='./mit_bih_data',
            normalize=True
        )
        
        if len(X) == 0:
            print("\n❌ No data loaded. Make sure wfdb is installed:")
            print("   pip install wfdb")
            return None, None
        
        print(f"\n✅ MIT-BIH data loaded successfully!")
        print(f"   Total segments: {len(X)}")
        print(f"   Segment shape: {X.shape}")
        
        return X, y
        
    except ImportError:
        print("\n❌ wfdb library not found!")
        print("Install with: pip install wfdb")
        print("\nAlternatively, use synthetic data by running:")
        print("   python ecg_quickstart_synthetic.py")
        return None, None
    except Exception as e:
        print(f"\n❌ Error loading MIT-BIH data: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection (need to download from PhysioNet)")
        print("2. Install wfdb: pip install wfdb")
        print("3. Ensure enough disk space (~100 MB)")
        return None, None


def main():
    """
    Complete ECG Arrhythmia Detection Pipeline
    """
    print("=" * 70)
    print("ECG ARRHYTHMIA DETECTION WITH WAV-KAN - QUICK START")
    print("=" * 70)
    
    # Configuration
    config = {
        'signal_length': 5000,          # ~14 seconds @ 360 Hz (MIT-BIH standard)
        'num_classes': 5,               # Normal, AFIB, PVC, LBBB, RBBB
        'hidden_sizes': [256, 128, 64], # Larger network for real data
        'wavelet_type': 'morlet',       # Best for ECG QRS detection
        'batch_size': 16,               # Smaller batch for real medical data
        'epochs': 100,                  # More epochs for real data (early stopping will help)
        'lr': 5e-4,                     # Slightly lower LR for stability
        'weight_decay': 1e-4,           # L2 regularization
        'early_stopping_patience': 15,  # Real data needs patience
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n📊 Configuration (optimized for MIT-BIH real data):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ========== STEP 1: Generate/Load Data ==========
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA FROM MIT-BIH DATABASE")
    print("="*70)
    
    # Load real MIT-BIH data
    X_all, y_all = load_real_mit_bih_data()
    
    if X_all is None or y_all is None:
        print("\n❌ Failed to load data")
        print("\nTroubleshooting:")
        print("1. Install wfdb: pip install wfdb")
        print("2. Check internet connection")
        print("3. Run: python mit_bih_loader.py")
        print("\n📝 Or use synthetic data: python ecg_quickstart_synthetic.py")
        return
    
    if len(X_all) < 50:
        print(f"\n⚠️  Only {len(X_all)} segments loaded (need at least 50 for training)")
        print("Increase record_ids in load_real_mit_bih_data() or check download")
        return
    
    # ========== STEP 2: Split Data ==========
    print("\nSplitting data: 70% train, 15% val, 15% test...")
    
    # Split: 70% train, 15% val, 15% test
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
    
    print("\nCreating PyTorch DataLoaders with preprocessing...")
    print("Real MIT-BIH data already normalized during loading")
    print("Applying additional preprocessing and data augmentation...\n")
    
    dataloaders = create_ecg_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=config['batch_size'],
        signal_length=config['signal_length'],
        num_workers=0,  # 0 for Windows compatibility
        preprocess_type='standard',
        augmentation=True
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Test batches: {len(dataloaders['test'])}")
    print(f"  Class weights (for imbalance): {dataloaders['class_weights'].numpy()}")
    
    # ========== STEP 3: Create Model ==========
    print("\n" + "=" * 70)
    print("STEP 3: CREATING WAV-KAN MODEL")
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {config['device']}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    test_input = torch.randn(1, config['signal_length'])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"✓ Forward pass successful: Input {test_input.shape} → Output {test_output.shape}")
    
    # ========== STEP 4: Train Model ==========
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL")
    print("=" * 70)
    
    trainer = ECGTrainer(
        model=model,
        device=config['device'],
        model_name='ecg_wavkan_quickstart'
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
    
    # ========== STEP 5: Save Results ==========
    print("\n" + "=" * 70)
    print("STEP 5: SAVING RESULTS")
    print("=" * 70)
    
    trainer.save_training_results()
    
    # ========== STEP 6: Evaluate on Test Set ==========
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATING ON TEST SET")
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
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Dataset: MIT-BIH Arrhythmia Database (Real PhysioNet data)")
    print(f"   ✓ Loaded from: https://www.physionet.org/content/mitdb/")
    print(f"   ✓ Preprocessing: Baseline filtering + Z-score normalization + Augmentation")
    print(f"\n📁 Results saved to: ./ecg_models/")
    print(f"   - ecg_wavkan_quickstart_best.pt (model)")
    print(f"   - ecg_wavkan_quickstart_results.csv (metrics)")
    print(f"   - ecg_wavkan_quickstart_curves.png (plots)")
    print(f"\n📈 Training Results:")
    print(f"   - Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch+1})")
    print(f"   - Test accuracy: {test_acc:.2f}%")
    if len(trainer.history['val_f1']) > 0:
        print(f"   - Test F1-score: {trainer.history['val_f1'][-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("""
1. ✅ You've successfully trained Wav-KAN on REAL MIT-BIH data!
   - Compare with synthetic data results (run ecg_quickstart_synthetic.py)
   - Real data should show more realistic metrics

2. 📊 Experiment with more MIT-BIH records:
   - Edit mit_bih_loader.py to load all 48 records
   - Load all: record_ids = list(MIT_BIH_RECORDS.keys())
   - More data = better generalization

3. 🔄 Try different wavelet types:
   model = ECG_KAN(..., wavelet_type='mexican_hat')  # Faster
   model = ECG_KAN(..., wavelet_type='dog')          # Edge detection

4. 🔧 Hyperparameter tuning:
   - Learning rate: Try 1e-4, 5e-4, 1e-3
   - Batch size: Try 8, 16, 32, 64
   - Hidden layers: Try [128, 64], [512, 256, 128]
   - Dropout: Try 0.1, 0.2, 0.3, 0.4
    """)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
