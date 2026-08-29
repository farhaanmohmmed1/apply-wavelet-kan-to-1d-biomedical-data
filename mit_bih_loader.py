'''
MIT-BIH Arrhythmia Database Data Loader
Downloads and processes real ECG data from PhysioNet
'''
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("⚠️  wfdb not installed. Run: pip install wfdb")



# Arrhythmia label mapping
# AAMI EC57 Standard Beat Mapping
AAMI_LABELS = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Non-ectopic (Normal)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic
    'V': 2, 'E': 2,                          # Ventricular ectopic
    'F': 3,                                  # Fusion beat
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced beat
}




def download_mit_bih_record(record_id, cache_dir='./mit_bih_data'):
    
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library not installed. Run: pip install wfdb")
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Loading MIT-BIH record {record_id}...")
        
        # wfdb automatically downloads from PhysioNet if not cached locally
        record = wfdb.rdrecord(f'mit-bih-arrhythmia-database/{record_id}', 
                              pn_dir='physionet.org/files/mitdb/1.0.0/',
                              cache_dir=str(cache_dir))
        
        # Read annotations (beat labels)
        annotation = wfdb.rdann(f'mit-bih-arrhythmia-database/{record_id}',
                               'atr',
                               pn_dir='physionet.org/files/mitdb/1.0.0/',
                               cache_dir=str(cache_dir))
        
        # Extract ECG signal (first channel - lead II which is standard)
        signal = record.p_signal[:, 0]
        sampling_rate = record.fs
        
        print(f"✓ Loaded record {record_id}")
        print(f"  Signal length: {len(signal)} samples")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration: {len(signal) / sampling_rate:.1f} seconds")
        print(f"  Beats: {len(annotation.sample)}")
        
        return signal, sampling_rate, annotation
        
    except Exception as e:
        print(f"❌ Error loading record {record_id}: {e}")
        return None, None, None

def extract_beat_windows(signal, annotation, window_size=250):
    """
    Extracts fixed-length beat segments centered on annotated R-peaks.
    250 samples @ 360 Hz ≈ 0.69 seconds window around the peak.
    """
    half_w = window_size // 2
    beats = []
    labels = []
    
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in AAMI_LABELS:
            # Check bounds to ensure the full window fits within the signal
            if half_w <= sample < (len(signal) - half_w):
                beat = signal[sample - half_w : sample + half_w]
                
                # Z-score normalization per beat
                std = beat.std()
                if std > 0:
                    beat = (beat - beat.mean()) / std
                    
                beats.append(beat)
                labels.append(AAMI_LABELS[symbol])
                
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

def load_mit_bih_dataset(record_ids=None, window_size=250, cache_dir='./mit_bih_data'):
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb not available. Install with: pip install wfdb")
    
    if record_ids is None:
        # Standard benchmark selection subset
        record_ids = ['100', '101', '103', '105', '106', '115', '118', '119', '200', '201']
    
    all_beats = []
    all_labels = []
    
    for record_id in record_ids:
        signal, fs, annotation = download_mit_bih_record(record_id, cache_dir)
        
        if signal is None:
            continue
            
        beats, labels = extract_beat_windows(signal, annotation, window_size=window_size)
        
        if len(beats) > 0:
            all_beats.append(beats)
            all_labels.append(labels)
            
    X = np.concatenate(all_beats, axis=0) if all_beats else np.array([])
    y = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    return X, y


def main_demo():
    """
    Demo: Load a few MIT-BIH records and show statistics
    """
    print("\n" + "="*70)
    print("MIT-BIH ARRHYTHMIA DATABASE LOADER - DEMO")
    print("="*70)
    
    # Install wfdb if needed
    if not WFDB_AVAILABLE:
        print("\n❌ wfdb library not found")
        print("Install with: pip install wfdb")
        print("\nThen run: python mit_bih_loader.py")
        return
    
    # Load a subset of records (quick demo)
    print("\nLoading 5 MIT-BIH records for demo...")
    print("(First run will download ~100 MB from PhysioNet)")
    
    record_ids = ['100', '101', '103', '115', '117']  # Mix of normal and arrhythmias
    
    try:
        X, y = load_mit_bih_dataset(
            record_ids=record_ids,
            segment_length=5000,
            sampling_rate=360,
            cache_dir='./mit_bih_data',
            normalize=True
        )
        
        print("\n✓ Data loaded successfully!")
        print(f"Shape: {X.shape}")
        print(f"Data type: {X.dtype}")
        print(f"Sample signal range: [{X.min():.3f}, {X.max():.3f}]")
        
        # Save for later use
        print("\n💾 Saving data...")
        np.save('mit_bih_signals.npy', X)
        np.save('mit_bih_labels.npy', y)
        print("✓ Saved as mit_bih_signals.npy and mit_bih_labels.npy")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_demo()
