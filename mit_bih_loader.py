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


# MIT-BIH Record IDs and their characteristics
MIT_BIH_RECORDS = {
    '100': 'Normal',
    '101': 'AFIB',
    '102': 'Normal',
    '103': 'AFIB',
    '104': 'Normal',
    '105': 'AFIB',
    '106': 'Normal',
    '107': 'AFIB',
    '108': 'Normal',
    '109': 'Normal',
    '111': 'Normal',
    '112': 'AFIB',
    '113': 'AFIB',
    '114': 'AFIB',
    '115': 'PVC',
    '116': 'Normal',
    '117': 'PVC',
    '118': 'PVC',
    '119': 'PVC',
    '121': 'Normal',
    '122': 'Normal',
    '123': 'PVC',
    '124': 'Normal'
}

# Arrhythmia label mapping
ARRHYTHMIA_LABELS = {
    'N': 0,  # Normal beat
    'L': 1,  # Left bundle branch block beat
    'R': 2,  # Right bundle branch block beat
    'A': 3,  # Atrial premature beat (AFIB indicator)
    'V': 4,  # Premature ventricular contraction (PVC)
    '!': 5,  # Ventricular flutter wave
}

# Simplified: 5 main classes
SIMPLIFIED_LABELS = {
    'N': 0,  # Normal
    'A': 1,  # Atrial fibrillation
    'V': 2,  # Premature Ventricular Contraction
    'L': 3,  # Left bundle branch block
    'R': 4,  # Right bundle branch block
}


def download_mit_bih_record(record_id, cache_dir='./mit_bih_data'):
    """
    Download MIT-BIH record from PhysioNet
    
    Args:
        record_id: Record number (e.g., '100', '101')
        cache_dir: Directory to cache downloaded files
    
    Returns:
        Tuple of (signal, sampling_rate, annotations) or None if download fails
    """
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


def segment_ecg_signal(signal, annotation, segment_length=5000, 
                       sampling_rate=360, overlap=0.5):
    """
    Segment long ECG record into fixed-length windows
    Each segment gets the most common beat label
    
    Args:
        signal: 1D ECG array
        annotation: wfdb annotation object with beat labels
        segment_length: Samples per segment (default: 5000 ≈ 14 seconds @ 360Hz)
        sampling_rate: Sampling rate in Hz
        overlap: Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        segments: [num_segments, segment_length]
        labels: [num_segments]
    """
    segments = []
    labels = []
    
    # Create beat-level labels array
    beat_labels = np.full(len(signal), 'N', dtype=object)
    for i, sample in enumerate(annotation.sample):
        if sample < len(beat_labels):
            beat_labels[sample] = annotation.symbol[i]
    
    # Segment with overlap
    step = int(segment_length * (1 - overlap))
    
    for start in range(0, len(signal) - segment_length, step):
        end = start + segment_length
        segment = signal[start:end]
        
        # Get most common label in this segment
        segment_beat_labels = beat_labels[start:end]
        
        # Count non-'N' beats
        label_counts = {}
        for label in segment_beat_labels:
            if label != 'N':  # Prioritize non-normal beats
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Get most common beat (or 'N' if all normal)
        if label_counts:
            most_common_label = max(label_counts, key=label_counts.get)
        else:
            most_common_label = 'N'
        
        # Map to class index
        if most_common_label in SIMPLIFIED_LABELS:
            label = SIMPLIFIED_LABELS[most_common_label]
        else:
            label = 0  # Default to normal
        
        segments.append(segment)
        labels.append(label)
    
    return np.array(segments, dtype=np.float32), np.array(labels)


def load_mit_bih_dataset(record_ids=None, segment_length=5000, 
                         sampling_rate=360, cache_dir='./mit_bih_data',
                         normalize=True):
    """
    Load multiple MIT-BIH records and segment them
    
    Args:
        record_ids: List of record IDs to load (e.g., ['100', '101', '102'])
                   If None, loads all available records
        segment_length: Samples per segment
        sampling_rate: Expected sampling rate (MIT-BIH uses 360 Hz)
        cache_dir: Directory for caching
        normalize: Whether to normalize signals
    
    Returns:
        Tuple of (X, y) where:
            X: [num_segments, segment_length]
            y: [num_segments] with class labels
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb not available. Install with: pip install wfdb")
    
    if record_ids is None:
        record_ids = list(MIT_BIH_RECORDS.keys())
    
    print(f"\n{'='*70}")
    print(f"Loading MIT-BIH Arrhythmia Database")
    print(f"{'='*70}")
    print(f"Records to load: {record_ids}")
    print(f"Target segment length: {segment_length} samples")
    print(f"Expected sampling rate: {sampling_rate} Hz")
    
    all_segments = []
    all_labels = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for record_id in record_ids:
        print(f"\n[{record_ids.index(record_id)+1}/{len(record_ids)}] Processing record {record_id}...")
        
        # Download and load
        signal, fs, annotation = download_mit_bih_record(record_id, cache_dir)
        
        if signal is None:
            print(f"⚠️  Skipping record {record_id}")
            continue
        
        # Resample if needed
        if fs != sampling_rate:
            from scipy import signal as sp_signal
            num_samples = int(len(signal) * sampling_rate / fs)
            signal = sp_signal.resample(signal, num_samples)
            print(f"  Resampled from {fs} Hz to {sampling_rate} Hz")
        
        # Segment
        segments, labels = segment_ecg_signal(
            signal, annotation,
            segment_length=segment_length,
            sampling_rate=sampling_rate,
            overlap=0.5
        )
        
        # Normalize
        if normalize:
            for i in range(len(segments)):
                mean = segments[i].mean()
                std = segments[i].std()
                if std > 0:
                    segments[i] = (segments[i] - mean) / std
        
        # Track class distribution
        for label in labels:
            class_counts[label] += 1
        
        all_segments.append(segments)
        all_labels.extend(labels)
        
        print(f"  Generated {len(segments)} segments")
        print(f"  Class distribution in record:")
        class_names = ['Normal', 'AFIB', 'PVC', 'LBBB', 'RBBB']
        for cls_id, count in class_counts.items():
            print(f"    {class_names[cls_id]}: {np.sum(np.array(all_labels) == cls_id)}")
    
    # Combine all segments
    X = np.vstack(all_segments) if all_segments else np.array([])
    y = np.array(all_labels)
    
    print(f"\n{'='*70}")
    print(f"Dataset Summary:")
    print(f"{'='*70}")
    print(f"Total segments: {len(X)}")
    print(f"Segment shape: {X.shape}")
    print(f"Label distribution:")
    for cls_id, name in enumerate(['Normal', 'AFIB', 'PVC', 'LBBB', 'RBBB']):
        count = np.sum(y == cls_id)
        percentage = 100 * count / len(y) if len(y) > 0 else 0
        print(f"  {name:12s}: {count:6d} ({percentage:5.1f}%)")
    print(f"{'='*70}\n")
    
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
