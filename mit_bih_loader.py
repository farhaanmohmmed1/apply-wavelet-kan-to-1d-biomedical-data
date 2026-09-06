"""
ecg_dataset_pipeline.py
========================
Full ECG dataset preparation pipeline for the MIT-BIH Arrhythmia Database,
built for training Wav-KAN (or any PyTorch 1D signal model).

This module extends `mit_bih_loader.py` (which downloads/reads individual
records) into a complete, leakage-safe data pipeline:

    download -> load full record -> segment into fixed windows
    -> label each window -> preprocess -> split by patient
    -> build PyTorch Datasets/DataLoaders -> class weights

Design choices are explained in the docstrings of each function. Defaults
are conservative (they favor "skip if unsure" over guessing a label), and
everything is pure Python + numpy/scipy/sklearn/wfdb/torch so it runs the
same way on Windows, macOS, and Linux.

Requires: wfdb, numpy, scipy, scikit-learn, torch (Python 3.10+)
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("wfdb not installed. Run: pip install wfdb")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("torch not installed. Run: pip install torch")


# ======================================================================
# 1. LABEL DEFINITIONS
# ======================================================================
# We classify each fixed-length window into ONE of 5 classes. Two different
# kinds of PhysioNet annotations feed into this:
#   - "beat" annotations (single symbol at an R-peak, e.g. 'N', 'V', 'L', 'R')
#   - "rhythm" annotations (aux_note text starting with '(', e.g. '(AFIB',
#     that mark the START of a rhythm segment and hold until the next one)
#
# AFIB is a *rhythm* label (it describes a stretch of time, not one beat),
# so it is handled separately from the beat-symbol classes.

CLASS_NAMES: List[str] = ["Normal", "AFIB", "PVC", "LBBB", "RBBB"]
CLASS_MAP: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Beat symbol -> class name. Only beats relevant to our 5 target classes are
# listed; anything else (paced beats, fusion beats, unreadable beats, etc.)
# is intentionally left unmapped and will cause that beat to be ignored
# during label assignment (per requirement: "skip invalid/low-quality
# windows" rather than guess).
BEAT_TO_CLASS: Dict[str, str] = {
    "N": "Normal",   # normal sinus beat
    "e": "Normal",   # atrial escape beat (grouped with Normal per AAMI)
    "j": "Normal",   # junctional escape beat (grouped with Normal per AAMI)
    "L": "LBBB",      # left bundle branch block beat
    "R": "RBBB",      # right bundle branch block beat
    "V": "PVC",       # premature ventricular contraction
}

# Rhythm aux_note text (without the leading '(' and trailing '\x00') -> class.
# Extend this dict if you later want more rhythm-based classes.
RHYTHM_TO_CLASS: Dict[str, str] = {
    "AFIB": "AFIB",
}

# A window is only labeled AFIB if at least this fraction of its samples
# fall inside an AFIB rhythm segment. Below this, we don't have a clean
# majority rhythm, so the beat-symbol path is used instead.
AFIB_COVERAGE_THRESHOLD = 0.5

# A window is only labeled by beat symbol if the single most common mapped
# beat symbol accounts for at least this fraction of ALL mapped beats found
# in the window. This avoids mislabeling windows that straddle two
# different beat types.
BEAT_MAJORITY_THRESHOLD = 0.6


# ======================================================================
# 2. DOWNLOAD / LOAD A FULL RECORD
# ======================================================================
def download_and_load_record(
    record_id: str, cache_dir: str = "./mit_bih_data"
) -> Tuple[Optional[np.ndarray], Optional[int], Optional["wfdb.Annotation"]]:
    """
    Download (if needed) and load one full MIT-BIH record.

    Unlike the beat-window demo loader, this keeps the ENTIRE signal
    (not just short snippets around annotations), since we need to slide
    a fixed window across the whole recording.

    wfdb's rdrecord()/rdann() have no 'download_dir' argument - only
    dl_database() actually writes files to a local folder. So we:
      1. Download the record + .atr annotation file into cache_dir
         (skipped if already cached).
      2. Read both back from the local path.

    Returns
    -------
    signal : np.ndarray, shape (n_samples,)
        Lead 0 (Modified Lead II for most mitdb records), raw physical units.
    fs : int
        Sampling rate in Hz (360 for all mitdb records).
    annotation : wfdb.Annotation
        Full annotation object (beat symbols + rhythm aux_notes).
    Returns (None, None, None) on failure so callers can skip bad records.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library not installed. Run: pip install wfdb")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    record_path = cache_dir / f"{record_id}.dat"

    try:
        if not record_path.exists():
            print(f"Downloading record {record_id}...")
            wfdb.dl_database(
                "mitdb",
                dl_dir=str(cache_dir),
                records=[str(record_id)],
                annotators=["atr"],
            )

        record = wfdb.rdrecord(record_name=str(cache_dir / str(record_id)))
        annotation = wfdb.rdann(
            record_name=str(cache_dir / str(record_id)), extension="atr"
        )

        signal = record.p_signal[:, 0].astype(np.float64)
        fs = int(record.fs)

        print(
            f"  Record {record_id}: {len(signal)} samples @ {fs} Hz "
            f"({len(signal) / fs / 60:.1f} min), {len(annotation.sample)} annotations"
        )
        return signal, fs, annotation

    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        return None, None, None


# ======================================================================
# 3. RHYTHM TIMELINE (needed for AFIB, since it's a rhythm, not a beat)
# ======================================================================
@dataclass
class RhythmSegment:
    start: int
    end: int          # exclusive
    label: Optional[str]  # e.g. 'AFIB', or None if not one of RHYTHM_TO_CLASS


def build_rhythm_timeline(
    annotation: "wfdb.Annotation", signal_length: int
) -> List[RhythmSegment]:
    """
    Turn wfdb's rhythm change-markers into a list of contiguous segments
    covering the whole record.

    In MIT-BIH annotation files, a rhythm change is marked by symbol '+'
    with the new rhythm name in aux_note, e.g. aux_note="(AFIB\\x00". That
    rhythm is in effect until the next '+' marker (or end of record).
    """
    changes = []
    for sample, symbol, aux in zip(
        annotation.sample, annotation.symbol, annotation.aux_note
    ):
        if symbol == "+":
            rhythm_name = aux.strip("\x00").lstrip("(")
            changes.append((int(sample), RHYTHM_TO_CLASS.get(rhythm_name)))

    if not changes:
        return [RhythmSegment(0, signal_length, None)]

    changes.sort(key=lambda x: x[0])
    segments = []
    for i, (start, label) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else signal_length
        segments.append(RhythmSegment(start, end, label))

    # Cover any gap before the first marker as "unknown rhythm"
    if segments[0].start > 0:
        segments.insert(0, RhythmSegment(0, segments[0].start, None))

    return segments


# ======================================================================
# 4. PREPROCESSING (applied to the FULL signal, before windowing)
# ======================================================================
def preprocess_full_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Baseline-wander removal via a high-pass Butterworth filter.

    Baseline wander (slow drift from breathing/electrode motion) lives
    below ~0.5 Hz, well below any ECG waveform content, so a gentle
    high-pass filter removes it without distorting QRS/P/T morphology.

    This is done on the FULL record (not per-window) so filtfilt has
    enough samples to settle and we don't introduce edge artifacts at
    every window boundary.

    Per-window z-score normalization happens later, in `segment_signal`,
    because normalizing per-window (rather than per-record) makes each
    window comparable regardless of a patient's baseline signal amplitude.
    """
    nyquist = fs / 2.0
    cutoff = 0.5 / nyquist  # 0.5 Hz high-pass
    b, a = butter(N=2, Wn=cutoff, btype="highpass")
    filtered = filtfilt(b, a, signal)
    return filtered.astype(np.float64)


def is_valid_window(window: np.ndarray, flatline_std: float = 1e-3) -> bool:
    """
    Reject windows that are unusable for training:
      - contain NaN/Inf (corrupt data)
      - are (near-)flatline, i.e. lead came loose or saturated
    Extend this with more checks (e.g. clipping detection) if your data
    turns out to need it.
    """
    if window.size == 0:
        return False
    if not np.all(np.isfinite(window)):
        return False
    if window.std() < flatline_std:
        return False
    return True


# ======================================================================
# 5. SEGMENTATION + LABEL ASSIGNMENT
# ======================================================================
def segment_signal(
    signal_length: int, window_size: int, overlap: float
) -> List[Tuple[int, int]]:
    """
    Compute (start, end) index pairs for sliding, overlapping windows.

    overlap=0.5 means each window shares half its samples with the next
    one (stride = window_size * (1 - overlap)).
    """
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"
    stride = max(1, int(window_size * (1 - overlap)))
    windows = []
    start = 0
    while start + window_size <= signal_length:
        windows.append((start, start + window_size))
        start += stride
    return windows


def assign_window_label(
    start: int,
    end: int,
    annotation: "wfdb.Annotation",
    rhythm_timeline: List[RhythmSegment],
) -> Optional[str]:
    """
    Decide the single class label for one window, using whichever
    annotation type gives a clean majority signal. Returns None if the
    window is ambiguous (caller should skip it).

    Step 1 - Rhythm vote (AFIB):
        Sum how many samples in [start, end) fall inside each rhythm
        segment. If an AFIB segment covers >= AFIB_COVERAGE_THRESHOLD of
        the window, label it AFIB immediately (rhythm labels take
        precedence because AFIB has no single defining beat symbol).

    Step 2 - Beat-symbol vote (Normal / PVC / LBBB / RBBB):
        Collect all beat annotations whose sample index falls inside the
        window and whose symbol maps to one of our classes. Take the most
        common mapped class; require it to be the majority
        (>= BEAT_MAJORITY_THRESHOLD of mapped beats) so a window that
        mixes e.g. Normal and PVC beats isn't force-labeled either way.

    If neither step produces a confident label, return None (skip window).
    """
    window_len = end - start

    # --- Step 1: rhythm-based label (AFIB) ---
    afib_samples = 0
    for seg in rhythm_timeline:
        if seg.label != "AFIB":
            continue
        overlap_start = max(start, seg.start)
        overlap_end = min(end, seg.end)
        if overlap_end > overlap_start:
            afib_samples += overlap_end - overlap_start
    if afib_samples / window_len >= AFIB_COVERAGE_THRESHOLD:
        return "AFIB"

    # --- Step 2: beat-symbol-based label ---
    mask = (annotation.sample >= start) & (annotation.sample < end)
    symbols_in_window = np.asarray(annotation.symbol)[mask]

    mapped = [BEAT_TO_CLASS[s] for s in symbols_in_window if s in BEAT_TO_CLASS]
    if not mapped:
        return None

    counts = Counter(mapped)
    dominant_class, dominant_count = counts.most_common(1)[0]
    if dominant_count / len(mapped) >= BEAT_MAJORITY_THRESHOLD:
        return dominant_class

    return None  # mixed/ambiguous window - skip


# ======================================================================
# 6. PER-RECORD DATASET BUILDER
# ======================================================================
def build_windows_for_record(
    record_id: str,
    cache_dir: str = "./mit_bih_data",
    window_size: int = 5000,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full per-record pipeline: download -> load -> preprocess -> segment ->
    label -> validate -> normalize.

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size), float32
    y : np.ndarray, shape (n_windows,), int64  (values are CLASS_MAP indices)
    Both empty arrays if the record failed to load or produced no valid,
    labeled windows.
    """
    signal, fs, annotation = download_and_load_record(record_id, cache_dir)
    if signal is None:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    signal = preprocess_full_signal(signal, fs)
    rhythm_timeline = build_rhythm_timeline(annotation, len(signal))
    candidate_windows = segment_signal(len(signal), window_size, overlap)

    X, y = [], []
    for start, end in candidate_windows:
        label_name = assign_window_label(start, end, annotation, rhythm_timeline)
        if label_name is None:
            continue  # ambiguous/unlabeled window - skip

        segment = signal[start:end]
        if not is_valid_window(segment):
            continue  # flatline/corrupt window - skip

        # Per-window z-score normalization
        std = segment.std()
        segment = (segment - segment.mean()) / std if std > 0 else segment

        X.append(segment.astype(np.float32))
        y.append(CLASS_MAP[label_name])

    if not X:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(X), np.asarray(y, dtype=np.int64)


# ======================================================================
# 7. MULTI-RECORD, PATIENT-SAFE SPLIT
# ======================================================================
@dataclass
class SplitResult:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_weights: "torch.Tensor" = field(default=None)


def build_dataset(
    train_records: Sequence[str],
    val_records: Sequence[str],
    test_records: Sequence[str],
    cache_dir: str = "./mit_bih_data",
    window_size: int = 5000,
    overlap: float = 0.5,
) -> SplitResult:
    """
    Build train/val/test arrays with NO patient overlap between splits.

    Splitting by record (not by randomly shuffling windows) is essential:
    consecutive/overlapping windows from the same patient are highly
    correlated, so a random split would leak near-duplicate windows
    across train/val/test and give an overly optimistic accuracy that
    won't hold up on genuinely new patients.

    `train_records`, `val_records`, and `test_records` must be disjoint
    sets of record IDs - this function does not check for you, so double
    check your lists (see `suggest_patient_split` below for a starting
    point).
    """
    overlap_check = set(train_records) & set(val_records) | set(train_records) & set(
        test_records
    ) | set(val_records) & set(test_records)
    if overlap_check:
        raise ValueError(
            f"Record IDs appear in more than one split (patient leakage risk): {overlap_check}"
        )

    def _build(records):
        Xs, ys = [], []
        for rec in records:
            X, y = build_windows_for_record(rec, cache_dir, window_size, overlap)
            if len(X):
                Xs.append(X)
                ys.append(y)
        if not Xs:
            return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.concatenate(Xs), np.concatenate(ys)

    print(f"\nBuilding TRAIN split from records: {list(train_records)}")
    X_train, y_train = _build(train_records)
    print(f"Building VAL split from records: {list(val_records)}")
    X_val, y_val = _build(val_records)
    print(f"Building TEST split from records: {list(test_records)}")
    X_test, y_test = _build(test_records)

    class_weights = compute_class_weights(y_train) if len(y_train) else None

    _print_class_distribution("train", y_train)
    _print_class_distribution("val", y_val)
    _print_class_distribution("test", y_test)

    return SplitResult(X_train, y_train, X_val, y_val, X_test, y_test, class_weights)


def _print_class_distribution(split_name: str, y: np.ndarray) -> None:
    counts = Counter(y.tolist())
    total = len(y)
    print(f"  [{split_name}] {total} windows: " + ", ".join(
        f"{CLASS_NAMES[c]}={counts.get(c, 0)}" for c in range(len(CLASS_NAMES))
    ))


def validate_split(
    split: "SplitResult",
    min_examples_per_class: int = 20,
    max_single_class_fraction: float = 0.98,
) -> bool:
    """
    Go/no-go check: run this BEFORE handing data to a trainer.

    Flags two failure modes that run silently otherwise:
      1. A class with too few examples in any split to learn/evaluate
         (e.g. 4 PVC windows in train - technically non-zero, still useless).
      2. A split that's almost entirely one class (e.g. a "validation set"
         that's 100% LBBB) - metrics on it would be meaningless regardless
         of how the model performs.

    Returns True if all checks pass, False (with printed reasons) otherwise.
    Doesn't raise, so you can inspect `split` interactively either way.
    """
    ok = True
    for name, y in [("train", split.y_train), ("val", split.y_val), ("test", split.y_test)]:
        if len(y) == 0:
            print(f"[FAIL] {name} split is empty.")
            ok = False
            continue

        counts = Counter(y.tolist())
        total = len(y)

        dominant_class, dominant_count = counts.most_common(1)[0]
        dominant_frac = dominant_count / total
        if dominant_frac > max_single_class_fraction:
            print(
                f"[FAIL] {name} split is {dominant_frac:.0%} {CLASS_NAMES[dominant_class]} "
                f"- metrics on this split won't mean much."
            )
            ok = False

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            n = counts.get(cls_idx, 0)
            if n == 0:
                print(f"[WARN] {name} split has 0 examples of '{cls_name}'.")
            elif n < min_examples_per_class:
                print(
                    f"[WARN] {name} split has only {n} examples of '{cls_name}' "
                    f"(< {min_examples_per_class}) - treat that class's metrics "
                    f"on this split with caution."
                )

    if ok:
        print("[OK] Split passed basic sanity checks.")
    else:
        print(
            "[STOP] Fix the record selection above before training - "
            "these aren't warnings you can train through."
        )
    return ok


# ======================================================================
# 8. CLASS IMBALANCE HANDLING
# ======================================================================
def compute_class_weights(y: np.ndarray) -> "torch.Tensor":
    """
    Inverse-frequency class weights for use with e.g.
    nn.CrossEntropyLoss(weight=class_weights).

    Classes with fewer windows (AFIB, LBBB, RBBB are usually rarer than
    Normal in MIT-BIH) get a higher weight so the loss doesn't just learn
    to predict "Normal" for everything.
    """
    from sklearn.utils.class_weight import compute_class_weight

    present_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y
    )
    # Map back onto the FULL class list (a class missing from y gets weight 0,
    # since the model will never see it in training - not because it doesn't matter)
    full_weights = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for cls, w in zip(present_classes, weights):
        full_weights[cls] = w

    if TORCH_AVAILABLE:
        return torch.tensor(full_weights, dtype=torch.float32)
    return full_weights


# ======================================================================
# 9. OPTIONAL AUGMENTATION (train split only)
# ======================================================================
def augment_segment(
    segment: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Light augmentation for training windows: small additive Gaussian noise
    + small random amplitude scaling. Keep this OFF for val/test - it
    should only ever be applied to training data, and only if minority
    classes need more effective examples.
    """
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    noise = rng.normal(0, noise_std, size=segment.shape)
    return (segment * scale + noise).astype(np.float32)


# ======================================================================
# 10. PYTORCH DATASET / DATALOADER
# ======================================================================
if TORCH_AVAILABLE:

    class ECGWindowDataset(Dataset):
        """
        Thin wrapper so X (n, window_size) / y (n,) numpy arrays plug
        straight into a standard PyTorch DataLoader and your ECGTrainer.

        Returns (signal, label) where signal has shape (1, window_size) -
        the channel-first format expected by 1D conv / Wav-KAN layers.
        """

        def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            augment: bool = False,
        ):
            self.X = X
            self.y = y
            self.augment = augment

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int):
            segment = self.X[idx]
            if self.augment:
                segment = augment_segment(segment)
            signal = torch.from_numpy(segment).float().unsqueeze(0)  # (1, window_size)
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return signal, label

    def build_dataloaders(
        split: SplitResult,
        batch_size: int = 64,
        augment_train: bool = False,
        use_weighted_sampler: bool = True,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build train/val/test DataLoaders.

        Two independent ways to handle imbalance are available - use
        ONE, not both, to avoid double-correcting:
          - `use_weighted_sampler=True`: oversample rare classes so each
            training batch is roughly balanced (works with any loss fn).
          - or set `use_weighted_sampler=False` and instead pass
            `split.class_weights` into nn.CrossEntropyLoss(weight=...).

        num_workers=0 is the safe default on Windows (Windows requires
        the DataLoader worker code to be guarded by
        `if __name__ == "__main__":`, which is easy to forget mid-notebook).
        Increase it once your training script has that guard in place.
        """
        train_ds = ECGWindowDataset(split.X_train, split.y_train, augment=augment_train)
        val_ds = ECGWindowDataset(split.X_val, split.y_val, augment=False)
        test_ds = ECGWindowDataset(split.X_test, split.y_test, augment=False)

        if use_weighted_sampler and len(split.y_train):
            class_sample_counts = np.bincount(split.y_train, minlength=len(CLASS_NAMES))
            sample_weights = 1.0 / np.maximum(class_sample_counts[split.y_train], 1)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader


# ======================================================================
# 11. HELPER: DISCOVER WHICH RECORDS CONTAIN WHICH LABELS
# ======================================================================
def discover_available_labels(
    record_ids: Sequence[str], cache_dir: str = "./mit_bih_data"
) -> Dict[str, Dict[str, int]]:
    """
    Scan a list of records and report how many beats/rhythm-segments of
    each target class they contain. Run this BEFORE committing to a
    train/val/test split - it tells you empirically which records
    actually carry AFIB/LBBB/RBBB annotations in your installed copy of
    the database, rather than relying on a hardcoded list that may be
    stale or mis-remembered.

    Returns {record_id: {class_name: raw_annotation_count}}
    """
    report = {}
    for rec in record_ids:
        signal, fs, annotation = download_and_load_record(rec, cache_dir)
        if signal is None:
            continue
        rhythm_timeline = build_rhythm_timeline(annotation, len(signal))

        beat_counts = Counter(
            BEAT_TO_CLASS[s] for s in annotation.symbol if s in BEAT_TO_CLASS
        )
        afib_samples = sum(
            seg.end - seg.start for seg in rhythm_timeline if seg.label == "AFIB"
        )
        counts = dict(beat_counts)
        if afib_samples > 0:
            counts["AFIB"] = afib_samples  # reported in samples, not beat count
        report[rec] = counts
        print(f"  {rec}: {counts}")
    return report


# ======================================================================
# 12. SUGGESTED PATIENT-LEVEL SPLIT
# ======================================================================
def suggest_patient_split() -> Tuple[List[str], List[str], List[str]]:
    """
    A commonly-cited inter-patient split for MIT-BIH (DS1/DS2, after
    de Chazal et al. 2004), widely reused in ECG-classification papers
    specifically because it separates patients cleanly between train and
    test. We further hold out a handful of DS1 records as a validation
    set so DS2 stays untouched until final evaluation.

    IMPORTANT: verify this against your actual research goals and, ideally,
    against the source paper / PhysioNet docs before publishing results -
    treat this as a reasonable, literature-grounded starting point rather
    than ground truth copied from an authoritative registry.
    """
    ds1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
           "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
           "223", "230"]
    ds2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
           "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
           "233", "234"]

    val_records = ds1[-4:]        # small held-out slice of DS1 for validation
    train_records = ds1[:-4]
    test_records = ds2
    return train_records, val_records, test_records


# ======================================================================
# 13. DEMO ENTRY POINT
# ======================================================================
def main_demo():
    print("\n" + "=" * 70)
    print("ECG DATASET PIPELINE - DEMO (small record subset)")
    print("=" * 70)

    if not (WFDB_AVAILABLE and TORCH_AVAILABLE):
        print("Missing dependency - install wfdb and torch first.")
        return

    # Real inter-patient split (DS1/DS2, see suggest_patient_split docstring).
    # Swap back to a 2-3 record subset only if you're smoke-testing a code
    # change and want a fast iteration loop.
    train_records, val_records, test_records = suggest_patient_split()

    split = build_dataset(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        window_size=5000,
        overlap=0.5,
    )

    if not validate_split(split):
        print("\nAborting before building DataLoaders - see [FAIL]/[WARN] above.")
        return

    train_loader, val_loader, test_loader = build_dataloaders(
        split, batch_size=32, use_weighted_sampler=True
    )

    xb, yb = next(iter(train_loader))
    print(f"\nSample batch: X {tuple(xb.shape)}, y {tuple(yb.shape)}")
    print(f"Class weights: {split.class_weights}")

    np.save("X_train.npy", split.X_train)
    np.save("y_train.npy", split.y_train)
    np.save("X_val.npy", split.X_val)
    np.save("y_val.npy", split.y_val)
    np.save("X_test.npy", split.X_test)
    np.save("y_test.npy", split.y_test)
    print("\nSaved train/val/test arrays as .npy files.")


if __name__ == "__main__":
    main_demo()