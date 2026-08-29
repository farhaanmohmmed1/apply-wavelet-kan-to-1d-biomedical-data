'''
Training script for Wav-KAN ECG Arrhythmia Detection
Handles training, validation, and comprehensive evaluation
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from ecg_kan import ECG_KAN
from ecg_data_loader import create_ecg_dataloaders


class ECGTrainer:
    """
    Trainer class for ECG arrhythmia detection with Wav-KAN
    Handles training, validation, evaluation, and model saving
    """
    
    def __init__(self, model, device='cuda', model_name='ecg_wavkan', save_dir='./ecg_models'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_f1': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def create_optimizer(self, lr=1e-3, weight_decay=1e-4, optimizer_type='adamw'):
        """
        Create optimizer with appropriate settings for ECG training
        
        Args:
            lr: Learning rate
            weight_decay: L2 regularization
            optimizer_type: 'adamw', 'adam', 'sgd'
        """
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            signals = batch['signal'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(signals)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': f'{total_loss/total:.4f}', 
                                     'acc': f'{100*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader, criterion):
        """
        Validate model and compute detailed metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{total_loss/total:.4f}', 
                                         'acc': f'{100*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        # Calculate additional metrics
        sensitivity, specificity, f1 = self._compute_metrics(all_labels, all_preds)
        
        return avg_loss, avg_acc, sensitivity, specificity, f1
    
    @staticmethod
    def _compute_metrics(y_true, y_pred):
        """
        Compute medical evaluation metrics for arrhythmia detection
        
        For multi-class: macro-average
        """
        from sklearn.metrics import recall_score, f1_score, precision_score
        
        # Sensitivity = Recall (important for detecting arrhythmias!)
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Specificity = TNR (for each class macro-averaged)
        # specificities = []
        # for c in range(max(y_true) + 1):
        #     y_true_binary = np.array(y_true) == c
        #     y_pred_binary = np.array(y_pred) == c
        #     tn = np.sum((~y_true_binary) & (~y_pred_binary))
        #     fp = np.sum((~y_true_binary) & (y_pred_binary))
        #     spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        #     specificities.append(spec)
        # specificity = np.mean(specificities)
        specificity = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        # F1 score (harmonic mean of precision and recall)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return sensitivity, specificity, f1
    
    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, 
              weight_decay=1e-4, class_weights=None, early_stopping_patience=15):
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: L2 regularization
            class_weights: Weights for imbalanced classes
            early_stopping_patience: Patience for early stopping
        """
        print(f"\n{'='*60}")
        print(f"Training ECG_KAN for Arrhythmia Detection")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        # Create optimizer and loss function
        optimizer = self.create_optimizer(lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_sensitivity, val_specificity, val_f1 = self.validate(val_loader, criterion)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_sensitivity'].append(val_sensitivity)
            self.history['val_specificity'].append(val_specificity)
            self.history['val_f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Sensitivity: {val_sensitivity:.4f} | Specificity: {val_specificity:.4f} | F1: {val_f1:.4f}")
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch)
                print(f"✓ New best model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best epoch: {self.best_epoch+1} | Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        checkpoint_path = self.save_dir / f'{self.model_name}_best.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_training_results(self):
        """Save training results to CSV"""
        results_df = pd.DataFrame(self.history)
        results_path = self.save_dir / f'{self.model_name}_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # Save plot
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot training curves (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss curve
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Val')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy curve
            axes[0, 1].plot(self.history['train_acc'], label='Train')
            axes[0, 1].plot(self.history['val_acc'], label='Val')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Sensitivity & Specificity
            axes[1, 0].plot(self.history['val_sensitivity'], label='Sensitivity (Recall)')
            axes[1, 0].plot(self.history['val_specificity'], label='Specificity (Precision)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Sensitivity and Specificity')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # F1 Score
            axes[1, 1].plot(self.history['val_f1'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Validation F1 Score')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = self.save_dir / f'{self.model_name}_curves.png'
            plt.savefig(plot_path, dpi=150)
            print(f"Training curves saved to {plot_path}")
            plt.close()
        except ImportError:
            print("Matplotlib not installed - skipping plot generation")


def main_example():
    """
    Example training script for ECG arrhythmia detection
    """
    print("ECG Arrhythmia Detection with Wav-KAN")
    print("=" * 60)
    
    # ==================== STEP 1: Load Data ====================
    print("\nStep 1: Loading ECG data...")
    # Replace this with actual data loading from PhysioNet
    X_train = np.random.randn(500, 4800).astype(np.float32)  # 500 training samples
    y_train = np.random.randint(0, 5, 500)  # 5 arrhythmia classes
    
    X_val = np.random.randn(100, 4800).astype(np.float32)
    y_val = np.random.randint(0, 5, 100)
    
    X_test = np.random.randn(100, 4800).astype(np.float32)
    y_test = np.random.randint(0, 5, 100)
    
    # ==================== STEP 2: Create DataLoaders ====================
    print("Step 2: Creating data loaders...")
    dataloaders = create_ecg_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32,
        signal_length=5000,
        preprocess_type='standard',
        augmentation=True
    )
    
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Class weights: {dataloaders['class_weights'].numpy()}")
    
    # ==================== STEP 3: Create Model ====================
    print("\nStep 3: Creating Wav-KAN model...")
    model = ECG_KAN(
        input_size=5000,
        hidden_sizes=[256, 128, 64],
        num_classes=5,  # Normal, AFIB, PVC, LBBB, RBBB
        wavelet_type='morlet',
        dropout_rate=0.2
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== STEP 4: Training ====================
    print("\nStep 4: Training model...")
    trainer = ECGTrainer(model, device='cuda', model_name='ecg_wavkan_morlet')
    
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        class_weights=dataloaders['class_weights'],
        early_stopping_patience=10
    )
    
    # ==================== STEP 5: Save Results ====================
    print("\nStep 5: Saving results...")
    trainer.save_training_results()
    
    print("\n✓ Training complete! Check ./ecg_models/ for results.")


if __name__ == "__main__":
    main_example()
