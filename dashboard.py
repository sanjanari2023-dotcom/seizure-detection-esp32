import re
import sys
import time
import serial
import threading
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import joblib

# Optional EDF support for uploaded EEG files
try:
    import mne
except ImportError:
    mne = None

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QScrollArea,
    QPushButton,
    QFileDialog,
    QProgressBar
)

import pyqtgraph as pg


# ============================================================
# CONFIG
# ============================================================

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

HISTORY = 300
GRAPH_REFRESH_MS = 80

FINGER_THRESHOLD = 10000

# ------------------------------------------------------------
# HEART-RATE ALERT SETTINGS
# ------------------------------------------------------------
# These are configurable UI alert thresholds, not medical diagnosis limits.
HIGH_HR_THRESHOLD = 120.0
LOW_HR_THRESHOLD = 50.0
HR_ALERT_COOLDOWN_MS = 10000

# ------------------------------------------------------------
# FALL DETECTION SETTINGS (MPU6050 GYROSCOPE)
# ------------------------------------------------------------
# Gyroscope values are expected in degrees/second.
# A fall is flagged as a SUSPECTED FALL when the angular-speed
# magnitude stays above the threshold for a few samples.
# This is a rule-based demo, not a medically validated detector.
FALL_GYRO_THRESHOLD = 200.0
FALL_CONFIRM_SAMPLES = 1
FALL_ALERT_COOLDOWN_MS = 0
FALL_ALERT_DISPLAY_MS = 5000

# ------------------------------------------------------------
# SENSOR FUSION CALIBRATION
# ------------------------------------------------------------
CALIBRATION_DURATION_SEC = 10.0
FUSION_BIO_WEIGHT = 0.70
FUSION_HR_WEIGHT = 0.20
FUSION_SPO2_WEIGHT = 0.10
FUSION_THRESHOLD = 0.75
FUSION_BIO_MIN_PROBABILITY = 0.80

# Trained model created from the CHB-MIT training script.
TRAINED_MODEL_PATH = str(Path(__file__).resolve().parent / "seizure_model.pkl")
TRAINING_DATASET_DIR = Path.home() / "Downloads" / "chbmit"

# ============================================================
# SIMPLE STUDENT-PROJECT ML SETTINGS
# ============================================================
# This is a demonstration classifier, not a medical diagnostic model.
ML_WINDOW_SAMPLES = 256
ML_STEP_SAMPLES = 64
ML_CONFIRM_WINDOWS = 5
ML_THRESHOLD = 0.90

# ============================================================
# EEG / CHB-MIT SETTINGS
# ============================================================
# CHB-MIT recordings are normally sampled at 256 Hz.
SAMPLE_RATE = 256.0
WINDOW_SAMPLES = 256
STEP_SAMPLES = 64
POSITIVE_OVERLAP = 0.50

# Preferred CHB-MIT scalp-EEG channels. If none is present, the
# program automatically selects the first channel identified as EEG.
PREFERRED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

FEATURE_NAMES = [
    "std",
    "ptp",
    "rms",
    "diff_std",
    "line_length",
    "max_diff",
    "zero_crossings",
    "spectral_centroid",
    "high_freq_ratio",
    "spectral_flatness",
]


# ============================================================
# SERIAL BRIDGE
# ============================================================



def train_model_from_dataset(dataset_dir=None, model_path=None, progress_callback=None):
    """Train the CHB-MIT model and save seizure_model.pkl."""
    if dataset_dir is None:
        dataset_dir = TRAINING_DATASET_DIR
    dataset_dir = Path(dataset_dir)

    if model_path is None:
        model_path = Path(__file__).resolve().parent / "seizure_model.pkl"
    model_path = Path(model_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"CHB-MIT dataset folder not found: {dataset_dir}"
        )

    edf_files = sorted(dataset_dir.rglob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(
            f"No .edf files found inside: {dataset_dir}"
        )

    rng = np.random.default_rng(42)
    X, y, groups = [], [], []

    for number, edf_path in enumerate(edf_files, start=1):
        if progress_callback:
            progress_callback(
                f"Training: reading {number}/{len(edf_files)} - {edf_path.name}"
            )

        X_file, y_file, groups_file, _ = process_edf(edf_path, rng)
        X.extend(X_file)
        y.extend(y_file)
        groups.extend(groups_file)

    if not X:
        raise RuntimeError(
            "No labelled seizure windows were found. "
            "Make sure the EDF files have matching *.edf.seizures files."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training requires both Normal and Seizure windows."
        )

    unique_groups = np.unique(groups)

    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )
    else:
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split = max(1, int(len(indices) * 0.80))
        train_idx = indices[:split]
        test_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    if progress_callback:
        progress_callback("Training Random Forest...")

    model.fit(X_train_scaled, y_train)

    if len(X_test) > 0:
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (probabilities >= 0.50).astype(np.int32)
        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    else:
        accuracy = 0.0
        balanced_accuracy = 0.0

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "window_samples": WINDOW_SAMPLES,
        "step_samples": STEP_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "preferred_channels": PREFERRED_CHANNELS,
        "training_dataset": "CHB-MIT Scalp EEG Database",
        "positive_overlap": POSITIVE_OVERLAP,
        "note": (
            "Student project EEG pattern classifier. "
            "Not clinically validated and not a medical diagnosis."
        ),
    }

    joblib.dump(bundle, model_path)

    return {
        "model_path": str(model_path),
        "windows": len(X),
        "normal": int(np.sum(y == 0)),
        "seizure": int(np.sum(y == 1)),
        "subjects": len(unique_groups),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }



# ============================================================
# SAME FEATURE EXTRACTION USED BY THE DASHBOARD
# ============================================================

def eeg_features(signal):
    x = np.asarray(signal, dtype=np.float32)

    if len(x) < 16:
        return None

    x = x - np.mean(x)

    std = float(np.std(x))
    ptp = float(np.ptp(x))
    rms = float(np.sqrt(np.mean(x ** 2)))

    diff = np.diff(x)
    diff_std = float(np.std(diff))
    line_length = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff))) if len(diff) else 0.0

    signs = np.sign(x)
    zero_crossings = float(
        np.sum(signs[1:] != signs[:-1]) / len(x)
    )

    spectrum = np.abs(np.fft.rfft(x))

    if len(spectrum) > 2:
        power = spectrum[1:] ** 2
        total_power = float(np.sum(power) + 1e-8)
        freqs = np.linspace(0.0, 1.0, len(power))

        spectral_centroid = float(
            np.sum(freqs * power) / total_power
        )

        high_freq_ratio = float(
            np.sum(power[int(len(power) * 0.35):]) /
            total_power
        )

        spectral_flatness = float(
            np.exp(np.mean(np.log(power + 1e-8))) /
            (np.mean(power) + 1e-8)
        )
    else:
        spectral_centroid = 0.0
        high_freq_ratio = 0.0
        spectral_flatness = 0.0

    return np.array([
        std,
        ptp,
        rms,
        diff_std,
        line_length,
        max_diff,
        zero_crossings,
        spectral_centroid,
        high_freq_ratio,
        spectral_flatness,
    ], dtype=np.float32)


# ============================================================
# CHB-MIT SEIZURE ANNOTATION PARSER
# ============================================================

def parse_seizure_file(annotation_path):
    """Read CHB-MIT binary *.edf.seizures annotations.

    Returns a list of (start_seconds, end_seconds).
    Falls back to the official *-summary.txt file when available.
    """
    try:
        raw = Path(annotation_path).read_bytes()
    except Exception:
        return []

    marker = b"\x01\x00\x00\xec"
    pos = raw.find(marker)

    if pos >= 0:
        # CHB-MIT compact annotation records use:
        #   1 byte + 2 skipped + 1 byte   -> 2-byte start offset
        #   7 skipped + 1 byte            -> duration
        #   4 skipped bytes
        p = pos + len(marker)
        intervals = []
        current_time = 0

        while p < len(raw):
            if p + 1 >= len(raw):
                break

            first = raw[p]
            p += 1

            if p + 2 >= len(raw):
                break
            p += 2
            second = raw[p]
            p += 1

            if p + 7 >= len(raw):
                break
            p += 7
            duration = raw[p]
            p += 1

            if p + 4 > len(raw):
                break
            p += 4

            start_offset = (first << 8) | second

            if start_offset == 0 and duration == 0:
                continue

            start_sec = current_time + start_offset
            end_sec = start_sec + duration

            if end_sec > start_sec:
                intervals.append((float(start_sec), float(end_sec)))
                current_time = end_sec

        if intervals:
            return intervals

    # Reliable fallback: official CHB-MIT patient summary.
    try:
        prefix = Path(annotation_path).name.split("_")[0]
        summary_path = Path(annotation_path).parent / f"{prefix}-summary.txt"
        if not summary_path.exists():
            matches = list(Path(annotation_path).parent.glob("*-summary.txt"))
            summary_path = matches[0] if matches else None

        if summary_path and summary_path.exists():
            text = summary_path.read_text(encoding="utf-8", errors="ignore")
            edf_name = Path(annotation_path).name
            if edf_name.endswith(".seizures"):
                edf_name = edf_name[:-len(".seizures")]

            pattern = (
                r"File Name:\s*" + re.escape(edf_name) +
                r"(.*?)(?=\n\s*File Name:|\Z)"
            )
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                block = match.group(1)
                starts = re.findall(
                    r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*"
                    r"([0-9]+(?:\.[0-9]+)?)\s*seconds",
                    block, flags=re.I
                )
                ends = re.findall(
                    r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*"
                    r"([0-9]+(?:\.[0-9]+)?)\s*seconds",
                    block, flags=re.I
                )
                if not starts:
                    starts = re.findall(
                        r"Start\s+Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
                        block, flags=re.I
                    )
                if not ends:
                    ends = re.findall(
                        r"End\s+Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
                        block, flags=re.I
                    )

                intervals = []
                for a, b in zip(starts, ends):
                    a, b = float(a), float(b)
                    if b > a:
                        intervals.append((a, b))
                if intervals:
                    return intervals
    except Exception:
        pass

    return []

def find_annotation_file(edf_path):
    """
    Find the annotation file while allowing the original CHB-MIT naming.

    Expected:
        chb03_01.edf
        chb03_01.edf.seizures
    """
    candidates = [
        Path(str(edf_path) + ".seizures"),
        edf_path.with_suffix(".seizures"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# ============================================================
# CHANNEL SELECTION
# ============================================================

def choose_eeg_channel(raw):
    upper_to_original = {
        ch.strip().upper(): ch
        for ch in raw.ch_names
    }

    for preferred in PREFERRED_CHANNELS:
        if preferred.upper() in upper_to_original:
            return upper_to_original[preferred.upper()]

    # Prefer channels that look like EEG channels.
    eeg_candidates = []

    for ch in raw.ch_names:
        kind = raw.get_channel_types(picks=[ch])[0]
        if kind == "eeg":
            eeg_candidates.append(ch)

    if eeg_candidates:
        return eeg_candidates[0]

    # Final fallback.
    return raw.ch_names[0]


# ============================================================
# WINDOW LABELING
# ============================================================

def window_overlaps_seizure(
    start_sec,
    end_sec,
    seizure_intervals,
    required_fraction=0.50,
):
    duration = end_sec - start_sec

    for seizure_start, seizure_end in seizure_intervals:
        overlap = max(
            0.0,
            min(end_sec, seizure_end) -
            max(start_sec, seizure_start)
        )

        if duration > 0 and overlap / duration >= required_fraction:
            return True

    return False


# ============================================================
# PROCESS ONE EDF
# ============================================================

def process_edf(
    edf_path,
    rng,
):
    annotation_path = find_annotation_file(edf_path)

    if annotation_path is None:
        print(
            f"  SKIP: no seizure annotation file for {edf_path.name}"
        )
        return [], [], [], []

    seizure_intervals = parse_seizure_file(annotation_path)

    if not seizure_intervals:
        print(
            f"  SKIP: could not parse seizure annotations in "
            f"{annotation_path.name}"
        )
        print(
            "  TIP: CHB-MIT .edf.seizures files are binary; "
            "the built-in parser and summary.txt fallback were used."
        )
        return [], [], [], []

    print(f"\nProcessing: {edf_path}")

    try:
        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            verbose="ERROR",
        )
    except Exception as exc:
        print(f"  ERROR reading EDF: {exc}")
        return [], [], [], []

    try:
        channel = choose_eeg_channel(raw)

        signal = raw.get_data(
            picks=[channel]
        )[0].astype(np.float32)

        sfreq = float(raw.info["sfreq"])

        # CHB-MIT is normally 256 Hz, but keep this robust.
        if abs(sfreq - SAMPLE_RATE) > 0.01:
            from scipy.signal import resample

            target_len = max(
                1,
                int(len(signal) * SAMPLE_RATE / sfreq)
            )

            signal = resample(
                signal,
                target_len
            ).astype(np.float32)

            sfreq = SAMPLE_RATE

        # Match the uploaded-file preprocessing in the dashboard.
        signal = signal - np.median(signal)

        scale = float(np.std(signal))

        if scale > 1e-8:
            signal = signal / scale

        window_seconds = WINDOW_SAMPLES / sfreq

        total_windows = (
            max(
                0,
                (len(signal) - WINDOW_SAMPLES) //
                STEP_SAMPLES + 1
            )
        )

        if total_windows == 0:
            print("  SKIP: recording is shorter than one window")
            return [], [], [], []

        positive_indices = []
        normal_indices = []

        for index in range(total_windows):
            start_sample = index * STEP_SAMPLES
            end_sample = start_sample + WINDOW_SAMPLES

            start_sec = start_sample / sfreq
            end_sec = start_sec + window_seconds

            positive = window_overlaps_seizure(
                start_sec,
                end_sec,
                seizure_intervals,
                POSITIVE_OVERLAP,
            )

            if positive:
                positive_indices.append(index)
            else:
                normal_indices.append(index)

        # Limit normal examples because CHB-MIT contains much more
        # normal EEG than seizure EEG.
        if len(normal_indices) > MAX_NORMAL_WINDOWS_PER_FILE:
            normal_indices = rng.choice(
                normal_indices,
                size=MAX_NORMAL_WINDOWS_PER_FILE,
                replace=False,
            ).tolist()

        if len(positive_indices) > MAX_SEIZURE_WINDOWS_PER_FILE:
            positive_indices = rng.choice(
                positive_indices,
                size=MAX_SEIZURE_WINDOWS_PER_FILE,
                replace=False,
            ).tolist()

        X_file = []
        y_file = []
        groups_file = []
        names_file = []

        for index in normal_indices:
            start = index * STEP_SAMPLES
            segment = signal[start:start + WINDOW_SAMPLES]

            features = eeg_features(segment)

            if features is not None and np.all(np.isfinite(features)):
                X_file.append(features)
                y_file.append(0)
                groups_file.append(edf_path.parent.name)
                names_file.append(edf_path.name)

        for index in positive_indices:
            start = index * STEP_SAMPLES
            segment = signal[start:start + WINDOW_SAMPLES]

            features = eeg_features(segment)

            if features is not None and np.all(np.isfinite(features)):
                X_file.append(features)
                y_file.append(1)
                groups_file.append(edf_path.parent.name)
                names_file.append(edf_path.name)

        print(
            f"  Channel: {channel} | "
            f"normal windows: {sum(np.array(y_file) == 0)} | "
            f"seizure windows: {sum(np.array(y_file) == 1)}"
        )

        return X_file, y_file, groups_file, names_file

    except Exception as exc:
        print(f"  ERROR processing {edf_path.name}: {exc}")
        return [], [], [], []





class SerialBridge(QObject):
    data_received = Signal(tuple)
    status_changed = Signal(str)


# ============================================================
# SERIAL THREAD
# ============================================================

class SerialReader(threading.Thread):

    def __init__(self, bridge):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.running = True
        self.ser = None

    def run(self):

        while self.running:

            try:

                self.bridge.status_changed.emit("CONNECTING")

                self.ser = serial.Serial(
                    SERIAL_PORT,
                    BAUD_RATE,
                    timeout=1
                )

                time.sleep(2)

                self.bridge.status_changed.emit("CONNECTED")

                while self.running:

                    line = (
                        self.ser.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )

                    if not line:
                        continue

                    if line.startswith("#"):
                        continue

                    parts = line.split(",")

                    if len(parts) != 13:
                        continue

                    try:

                        millis = int(parts[0])

                        hr = float(parts[1])
                        spo2 = int(parts[2])

                        ir = int(parts[3])
                        red = int(parts[4])

                        ax = float(parts[5])
                        ay = float(parts[6])
                        az = float(parts[7])

                        gx = float(parts[8])
                        gy = float(parts[9])
                        gz = float(parts[10])

                        bio_raw = int(parts[11])
                        bio_v = float(parts[12])

                    except ValueError:
                        continue

                    self.bridge.data_received.emit(
                        (
                            millis,
                            hr,
                            spo2,
                            ir,
                            red,
                            ax,
                            ay,
                            az,
                            gx,
                            gy,
                            gz,
                            bio_raw,
                            bio_v
                        )
                    )

            except serial.SerialException:

                self.bridge.status_changed.emit(
                    "DISCONNECTED"
                )

                time.sleep(2)

            except Exception as e:

                print("Serial error:", e)

                self.bridge.status_changed.emit(
                    "ERROR"
                )

                time.sleep(2)

    def stop(self):

        self.running = False

        try:
            if self.ser:
                self.ser.close()
        except:
            pass



# ============================================================
# SIMPLE EEG ML MODEL
# ============================================================
# For a student demonstration, this model is trained on
# synthetic "normal" and "seizure-like" EEG patterns.
#
# IMPORTANT:
# This does NOT diagnose epilepsy and should not be presented
# as a clinically validated seizure detector.
# Replace the synthetic training data with labelled EEG data
# (e.g. CHB-MIT) for a real ML experiment.
# ============================================================

def eeg_features(signal):
    """Extract simple features from one BioAmp window."""
    x = np.asarray(signal, dtype=np.float32)

    if len(x) < 16:
        return None

    # Remove DC/baseline
    x = x - np.mean(x)

    std = float(np.std(x))
    ptp = float(np.ptp(x))
    rms = float(np.sqrt(np.mean(x ** 2)))

    diff = np.diff(x)
    diff_std = float(np.std(diff))
    line_length = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff))) if len(diff) else 0.0

    # Zero crossings of the centred signal
    signs = np.sign(x)
    zero_crossings = float(np.sum(signs[1:] != signs[:-1]) / len(x))

    # Normalized spectral features
    spectrum = np.abs(np.fft.rfft(x))
    if len(spectrum) > 2:
        power = spectrum[1:] ** 2
        total_power = float(np.sum(power) + 1e-8)
        freqs = np.linspace(0.0, 1.0, len(power))

        spectral_centroid = float(
            np.sum(freqs * power) / total_power
        )

        high_freq_ratio = float(
            np.sum(power[int(len(power) * 0.35):]) / total_power
        )

        spectral_flatness = float(
            np.exp(np.mean(np.log(power + 1e-8))) /
            (np.mean(power) + 1e-8)
        )
    else:
        spectral_centroid = 0.0
        high_freq_ratio = 0.0
        spectral_flatness = 0.0

    return np.array([
        std,
        ptp,
        rms,
        diff_std,
        line_length,
        max_diff,
        zero_crossings,
        spectral_centroid,
        high_freq_ratio,
        spectral_flatness
    ], dtype=np.float32)


class DemoSeizureML:
    """
    Small Random Forest classifier for the student demonstration.

    It learns from synthetic EEG-like signals. This lets the UI
    run immediately without requiring a large EEG dataset.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            random_state=42,
            class_weight="balanced"
        )
        self.train_demo_model()

    def make_normal(self, n, rng):
        t = np.linspace(0, 1, n, endpoint=False)

        base = (
            0.50 * np.sin(2 * np.pi * rng.uniform(4, 10) * t) +
            0.25 * np.sin(2 * np.pi * rng.uniform(8, 16) * t) +
            0.12 * rng.normal(size=n)
        )

        # Slow baseline drift
        base += 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)

        return base

    def make_seizure_like(self, n, rng):
        t = np.linspace(0, 1, n, endpoint=False)

        # Larger rhythmic bursts with stronger high-frequency activity
        freq = rng.uniform(5, 12)

        burst = (
            1.25 * np.sin(2 * np.pi * freq * t) +
            0.70 * np.sin(2 * np.pi * (freq * 2.0) * t) +
            0.45 * np.sin(2 * np.pi * (freq * 3.0) * t)
        )

        envelope = 0.35 + 0.65 * (
            0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.4, 1.2) * t)
        )

        x = burst * envelope + 0.20 * rng.normal(size=n)

        # Add occasional sharp activity
        for _ in range(rng.integers(2, 6)):
            center = rng.integers(10, n - 10)
            width = rng.integers(2, 7)
            idx = np.arange(max(0, center - width), min(n, center + width))
            x[idx] += rng.uniform(1.0, 2.5)

        return x

    def train_demo_model(self):
        rng = np.random.default_rng(42)

        X = []
        y = []

        # Normal examples
        for _ in range(500):
            signal = self.make_normal(ML_WINDOW_SAMPLES, rng)
            X.append(eeg_features(signal))
            y.append(0)

        # Seizure-like examples
        for _ in range(500):
            signal = self.make_seizure_like(ML_WINDOW_SAMPLES, rng)
            X.append(eeg_features(signal))
            y.append(1)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        self.model.fit(X, y)

    def predict(self, signal):
        features = eeg_features(signal)

        if features is None:
            return 0.0, "WAITING"

        probability = float(
            self.model.predict_proba(
                features.reshape(1, -1)
            )[0][1]
        )

        if probability >= ML_THRESHOLD:
            return probability, "SEIZURE-LIKE"
        return probability, "NORMAL"



# ============================================================
# TRAINED EEG MODEL LOADER
# ============================================================

class TrainedEEGModel:
    """
    Loads the model produced by train_seizure_model.py.

    The training script must save a dictionary containing:
        model
        scaler
        feature_names
    """

    def __init__(self, model_path=TRAINED_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False

        try:
            import joblib
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.scaler = bundle.get("scaler")
            self.feature_names = bundle.get("feature_names")
            self.loaded = True
        except Exception:
            # Model is optional until the user trains it from CHB-MIT data.
            self.loaded = False

    def predict_window(self, signal):
        if not self.loaded:
            raise RuntimeError(
                f"Trained model not found: {self.model_path}. "
                "Run train_seizure_model.py first."
            )

        features = eeg_features(signal)
        if features is None:
            return 0.0, "WAITING"

        X = features.reshape(1, -1)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        probability = float(self.model.predict_proba(X)[0][1])

        if probability >= ML_THRESHOLD:
            return probability, "SEIZURE-LIKE"
        return probability, "NORMAL"


# ============================================================
# SEIZURE STATUS WIDGET
# ============================================================

class SeizureStatus(QFrame):

    def __init__(self):
        super().__init__()

        self.setObjectName("SeizureStatus")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(6)

        title = QLabel("🧠  SEIZURE DETECTION")
        title.setObjectName("SeizureTitle")

        self.status = QLabel("WAITING FOR EEG")
        self.status.setObjectName("SeizureState")
        self.status.setAlignment(Qt.AlignCenter)

        self.probability = QLabel("Fusion Score: --")
        self.probability.setObjectName("SeizureProbability")
        self.probability.setAlignment(Qt.AlignCenter)

        self.info = QLabel(
            "Press START to begin collecting BioAmp samples"
        )
        self.info.setObjectName("SeizureInfo")
        self.info.setAlignment(Qt.AlignCenter)

        self.disclaimer = QLabel(
            "Demo ML pattern detector • not a medical diagnosis"
        )
        self.disclaimer.setObjectName("SeizureDisclaimer")
        self.disclaimer.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(self.status)
        layout.addWidget(self.probability)
        layout.addWidget(self.info)
        layout.addWidget(self.disclaimer)



# ============================================================
# SENSOR SECTION
# ============================================================

class SensorSection(QFrame):

    def __init__(
        self,
        title,
        icon,
        unit="",
        graph_label="",
        line_color="#42A5F5",
        xyz=False
    ):

        super().__init__()

        self.xyz = xyz
        self.unit = unit

        self.setObjectName(
            "SensorSection"
        )

        self.setMinimumHeight(
            420
        )

        main_layout = QHBoxLayout(
            self
        )

        main_layout.setContentsMargins(
            28,
            24,
            28,
            24
        )

        main_layout.setSpacing(
            28
        )

        # ====================================================
        # LEFT DATA SIDE
        # ====================================================

        left = QVBoxLayout()

        left.setSpacing(
            10
        )

        title_label = QLabel(
            f"{icon}  {title}"
        )

        title_label.setObjectName(
            "SensorTitle"
        )

        left.addWidget(
            title_label
        )

        left.addStretch()

        # ----------------------------------------------------
        # SINGLE VALUE
        # ----------------------------------------------------

        if not xyz:

            self.value_label = QLabel(
                "--"
            )

            self.value_label.setObjectName(
                "MainValue"
            )

            self.value_label.setAlignment(
                Qt.AlignCenter
            )

            self.unit_label = QLabel(
                unit
            )

            self.unit_label.setObjectName(
                "MainUnit"
            )

            self.unit_label.setAlignment(
                Qt.AlignCenter
            )

            left.addWidget(
                self.value_label
            )

            left.addWidget(
                self.unit_label
            )

        # ----------------------------------------------------
        # XYZ VALUES
        # ----------------------------------------------------

        else:

            self.x_label = QLabel(
                f"X   0.000 {unit}"
            )

            self.y_label = QLabel(
                f"Y   0.000 {unit}"
            )

            self.z_label = QLabel(
                f"Z   0.000 {unit}"
            )

            for label in (
                self.x_label,
                self.y_label,
                self.z_label
            ):
                label.setObjectName(
                    "XYZLarge"
                )

            left.addWidget(
                self.x_label
            )

            left.addWidget(
                self.y_label
            )

            left.addWidget(
                self.z_label
            )

        left.addStretch()

        # info line
        self.info_label = QLabel(
            ""
        )

        self.info_label.setObjectName(
            "InfoText"
        )

        left.addWidget(
            self.info_label
        )

        # ====================================================
        # RIGHT GRAPH SIDE
        # ====================================================

        graph_box = QVBoxLayout()

        self.plot = pg.PlotWidget()

        self.plot.setBackground(
            "#15191F"
        )

        self.plot.showGrid(
            x=True,
            y=True,
            alpha=0.15
        )

        self.plot.setLabel(
            "bottom",
            "Samples"
        )

        if graph_label:

            self.plot.setLabel(
                "left",
                graph_label
            )

        self.plot.getAxis(
            "left"
        ).enableAutoSIPrefix(False)

        self.plot.getAxis(
            "bottom"
        ).enableAutoSIPrefix(False)

        self.plot.setMouseEnabled(
            x=False,
            y=False
        )

        self.plot.hideButtons()

        self.plot.setMinimumHeight(
            330
        )

        # ----------------------------------------------------
        # SINGLE GRAPH
        # ----------------------------------------------------

        if not xyz:

            self.line = self.plot.plot(
                pen=pg.mkPen(
                    color=line_color,
                    width=2.4
                )
            )

        # ----------------------------------------------------
        # XYZ GRAPH
        # ----------------------------------------------------

        else:

            self.plot.addLegend()

            self.line_x = self.plot.plot(
                pen=pg.mkPen(
                    "#42A5F5",
                    width=2.2
                ),
                name="X"
            )

            self.line_y = self.plot.plot(
                pen=pg.mkPen(
                    "#FFA726",
                    width=2.2
                ),
                name="Y"
            )

            self.line_z = self.plot.plot(
                pen=pg.mkPen(
                    "#66BB6A",
                    width=2.2
                ),
                name="Z"
            )

        graph_box.addWidget(
            self.plot
        )

        main_layout.addLayout(
            left,
            1
        )

        main_layout.addLayout(
            graph_box,
            3
        )


    def set_value(
        self,
        value,
        unit=None
    ):

        self.value_label.setText(
            value
        )

        if unit is not None:

            self.unit_label.setText(
                unit
            )


    def set_xyz(
        self,
        x,
        y,
        z
    ):

        self.x_label.setText(
            f"X   {x:+.3f} {self.unit}"
        )

        self.y_label.setText(
            f"Y   {y:+.3f} {self.unit}"
        )

        self.z_label.setText(
            f"Z   {z:+.3f} {self.unit}"
        )


# ============================================================
# MAIN DASHBOARD
# ============================================================

class Dashboard(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Multimodal Sensor System"
        )

        self.resize(
            1650,
            950
        )

        self.setMinimumSize(
            1200,
            750
        )

        # ====================================================
        # BUFFERS
        # ====================================================

        self.ir_data = deque(
            maxlen=HISTORY
        )

        self.red_data = deque(
            maxlen=HISTORY
        )

        self.bio_data = deque(
            maxlen=HISTORY
        )

        # ML EEG buffer
        self.ml_bio_buffer = deque(
            maxlen=ML_WINDOW_SAMPLES
        )

        self.ml_counter = 0
        self.seizure_positive_count = 0
        self.fall_candidate_count = 0
        self.fall_alert_active = False
        self.fall_candidate_count = 0
        self.fall_alert_active = False
        self.last_fall_alert_ms = 0

        # Student-demo ML model
        self.ml_model = DemoSeizureML()

        # --------------------------------------------------------
        # ACQUISITION CONTROL
        # --------------------------------------------------------
        # Serial data may arrive continuously, but it is ignored
        # until the user presses START.
        self.acquisition_active = False

        # --------------------------------------------------------
        # ALERT STATE
        # --------------------------------------------------------
        self.hr_alert_active = False
        self.last_hr_alert_ms = 0

        # Fall detection state
        self.fall_alert_active = False
        self.fall_candidate_count = 0
        self.last_fall_alert_ms = 0
        self.fall_armed = True

        # Latest values for the compact top summary
        self.latest_hr = None
        self.latest_spo2 = None
        self.latest_bio = None
        self.hr_alert_active = False
        self.last_hr_alert_ms = 0
        self.latest_accel = (None, None, None)
        self.latest_gyro = (None, None, None)

        # --------------------------------------------------------
        # SENSOR FUSION CALIBRATION STATE
        # --------------------------------------------------------
        self.calibration_active = False
        self.calibration_complete = False
        self.calibration_started_at = 0.0
        self.calibration_hr = []
        self.calibration_spo2 = []
        self.calibration_bio = []
        self.hr_baseline = None
        self.hr_baseline_std = None
        self.spo2_baseline = None
        self.spo2_baseline_std = None
        self.bio_baseline = None
        self.bio_baseline_std = None

        self.ax_data = deque(
            maxlen=HISTORY
        )

        self.ay_data = deque(
            maxlen=HISTORY
        )

        self.az_data = deque(
            maxlen=HISTORY
        )

        self.gx_data = deque(
            maxlen=HISTORY
        )

        self.gy_data = deque(
            maxlen=HISTORY
        )

        self.gz_data = deque(
            maxlen=HISTORY
        )

        self.selected_eeg_file = None
        self.eeg_regions = []

        self.build_ui()

        self.apply_style()

        # ====================================================
        # SERIAL
        # ====================================================

        self.bridge = SerialBridge()

        self.bridge.data_received.connect(
            self.process_data
        )

        self.bridge.status_changed.connect(
            self.update_status
        )

        self.reader = SerialReader(
            self.bridge
        )

        self.reader.start()

        # ====================================================
        # GRAPH TIMER
        # ====================================================

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.refresh_graphs
        )

        self.timer.start(
            GRAPH_REFRESH_MS
        )


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        root = QWidget()

        self.setCentralWidget(
            root
        )

        layout = QVBoxLayout(
            root
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout.setSpacing(
            0
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = QFrame()

        header.setObjectName(
            "Header"
        )

        header_layout = QHBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            32,
            18,
            32,
            18
        )

        left_header = QVBoxLayout()

        title = QLabel(
            "MULTIMODAL HEALTH MONITOR"
        )

        title.setObjectName(
            "MainTitle"
        )

        subtitle = QLabel(
            "ESP32 • MAX30102 • MPU6050 • BioAmp • ADS1115"
        )

        subtitle.setObjectName(
            "Subtitle"
        )

        left_header.addWidget(
            title
        )

        left_header.addWidget(
            subtitle
        )

        header_layout.addLayout(
            left_header
        )

        header_layout.addStretch()

        right_header = QVBoxLayout()

        self.status_label = QLabel(
            "● CONNECTING"
        )

        self.status_label.setObjectName(
            "Status"
        )

        self.status_label.setAlignment(
            Qt.AlignRight
        )

        port = QLabel(
            f"{SERIAL_PORT} • {BAUD_RATE} baud"
        )

        port.setObjectName(
            "Subtitle"
        )

        port.setAlignment(
            Qt.AlignRight
        )

        right_header.addWidget(
            self.status_label
        )

        right_header.addWidget(
            port
        )

        header_layout.addLayout(
            right_header
        )

        layout.addWidget(
            header
        )

        # ====================================================
        # ACQUISITION CONTROL / TOP VALUES
        # ====================================================

        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")

        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(24, 16, 24, 16)
        control_layout.setSpacing(12)

        control_row = QHBoxLayout()
        control_row.setSpacing(16)

        self.acquisition_label = QLabel("● NOT STARTED")
        self.acquisition_label.setObjectName("AcquisitionLabel")

        self.start_button = QPushButton("▶  START")
        self.start_button.setObjectName("StartButton")
        self.start_button.setMinimumHeight(48)
        self.start_button.setMinimumWidth(170)
        self.start_button.clicked.connect(self.toggle_acquisition)

        self.calibrate_button = QPushButton("⚙  CALIBRATE SENSORS")
        self.calibrate_button.setObjectName("CalibrateButton")
        self.calibrate_button.setMinimumHeight(48)
        self.calibrate_button.setMinimumWidth(210)
        self.calibrate_button.clicked.connect(self.start_calibration)

        control_row.addWidget(self.acquisition_label)
        control_row.addStretch()
        control_row.addWidget(self.calibrate_button)
        control_row.addWidget(self.start_button)

        control_layout.addLayout(control_row)

        # Compact live values shown at the top.
        values_row = QHBoxLayout()
        values_row.setSpacing(10)

        self.top_hr = self.make_top_value("♥ HR", "-- BPM")
        self.top_spo2 = self.make_top_value("O₂ SpO₂", "-- %")
        self.top_bio = self.make_top_value("🧠 BioAmp", "-- V")
        self.top_accel = self.make_top_value("↕ ACCEL", "X --  Y --  Z -- g")
        self.top_gyro = self.make_top_value("⟳ GYRO", "X --  Y --  Z -- °/s")

        for card in (
            self.top_hr,
            self.top_spo2,
            self.top_bio,
            self.top_accel,
            self.top_gyro,
        ):
            values_row.addWidget(card, 1)

        control_layout.addLayout(values_row)

        self.calibration_label = QLabel(
            "⚪ SENSOR CALIBRATION: NOT CALIBRATED — press CALIBRATE SENSORS while the person is resting"
        )
        self.calibration_label.setObjectName("CalibrationLabel")
        self.calibration_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.calibration_label)

        layout.addWidget(control_panel)

        # ====================================================
        # ALERT BANNER
        # ====================================================

        self.alert_banner = QLabel("✓  NO ACTIVE ALERT")
        self.alert_banner.setObjectName("AlertBanner")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setMinimumHeight(42)
        layout.addWidget(self.alert_banner)

        # Non-blocking toast notification.
        # It disappears automatically, so it does not interrupt the sensor UI.
        self.toast = QLabel("", self)
        self.toast.setObjectName("Toast")
        self.toast.setAlignment(Qt.AlignCenter)
        self.toast.hide()

        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(self.toast.hide)

        # Fall alerts are visible for exactly 5 seconds, then the banner
        # returns to the normal state automatically.
        self.fall_alert_timer = QTimer(self)
        self.fall_alert_timer.setSingleShot(True)
        self.fall_alert_timer.timeout.connect(self.clear_fall_alert)

        # ====================================================
        # SCROLL AREA
        # ====================================================

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        content = QWidget()

        content_layout = QVBoxLayout(
            content
        )

        content_layout.setContentsMargins(
            24,
            24,
            24,
            24
        )

        content_layout.setSpacing(
            20
        )

        # ====================================================
        # SEIZURE DETECTION / ML
        # ====================================================

        self.seizure_status = SeizureStatus()

        content_layout.addWidget(
            self.seizure_status
        )

        # ====================================================
        # UPLOAD / ANALYZE EEG FILE
        # ====================================================

        self.file_analysis_panel = QFrame()
        self.file_analysis_panel.setObjectName("FileAnalysisPanel")

        file_layout = QVBoxLayout(self.file_analysis_panel)
        file_layout.setContentsMargins(24, 18, 24, 18)
        file_layout.setSpacing(10)

        file_title = QLabel("📁  EEG FILE ANALYSIS")
        file_title.setObjectName("FileAnalysisTitle")
        file_layout.addWidget(file_title)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)

        self.file_path_label = QLabel("No EEG file selected")
        self.file_path_label.setObjectName("FilePathLabel")
        self.file_path_label.setWordWrap(True)

        self.choose_file_button = QPushButton("CHOOSE .EDF")
        self.choose_file_button.setObjectName("ChooseFileButton")
        self.choose_file_button.clicked.connect(self.choose_eeg_file)

        self.analyze_file_button = QPushButton("ANALYZE EEG")
        self.analyze_file_button.setObjectName("AnalyzeFileButton")
        self.analyze_file_button.clicked.connect(self.analyze_eeg_file)
        self.analyze_file_button.setEnabled(False)

        file_row.addWidget(self.file_path_label, 1)
        file_row.addWidget(self.choose_file_button)
        file_row.addWidget(self.analyze_file_button)

        file_layout.addLayout(file_row)

        self.file_result = QLabel(
            "Upload a CHB-MIT EDF recording to analyze it."
        )
        self.file_result.setObjectName("FileResult")
        self.file_result.setAlignment(Qt.AlignCenter)
        self.file_result.setWordWrap(True)

        file_layout.addWidget(self.file_result)

        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.setBackground("#15191F")
        self.eeg_plot.showGrid(x=True, y=True, alpha=0.15)
        self.eeg_plot.setLabel("bottom", "Time (seconds)")
        self.eeg_plot.setLabel("left", "EEG (normalized)")
        self.eeg_plot.setMinimumHeight(300)
        self.eeg_plot.setMouseEnabled(x=False, y=False)
        self.eeg_plot.hideButtons()
        self.eeg_line = self.eeg_plot.plot(
            pen=pg.mkPen(color="#42A5F5", width=1.5)
        )
        file_layout.addWidget(self.eeg_plot)

        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.hide()
        file_layout.addWidget(self.file_progress)

        content_layout.addWidget(self.file_analysis_panel)

        # ====================================================
        # HEART RATE
        # ====================================================

        self.hr_section = SensorSection(
            "HEART RATE",
            "♥",
            "BPM",
            "IR",
            "#EF5350"
        )

        content_layout.addWidget(
            self.hr_section
        )

        # ====================================================
        # SPO2
        # ====================================================

        self.spo2_section = SensorSection(
            "SpO₂",
            "O₂",
            "%",
            "RED",
            "#26C6DA"
        )

        content_layout.addWidget(
            self.spo2_section
        )

        # ====================================================
        # BIOAMP
        # ====================================================

        self.bio_section = SensorSection(
            "BIOAMP",
            "🧠",
            "V",
            "Voltage (V)",
            "#AB47BC"
        )

        self.bio_section.plot.setYRange(
            0,
            3.3,
            padding=0
        )

        content_layout.addWidget(
            self.bio_section
        )

        # ====================================================
        # MPU ACCEL
        # ====================================================

        self.accel_section = SensorSection(
            "MPU6050 ACCELERATION",
            "↕",
            "g",
            "g",
            xyz=True
        )

        self.accel_section.plot.setYRange(
            -2,
            2,
            padding=0
        )

        content_layout.addWidget(
            self.accel_section
        )

        # ====================================================
        # MPU GYRO
        # ====================================================

        self.gyro_section = SensorSection(
            "MPU6050 GYROSCOPE",
            "⟳",
            "°/s",
            "°/s",
            xyz=True
        )

        self.gyro_section.plot.setYRange(
            -50,
            50,
            padding=0
        )

        content_layout.addWidget(
            self.gyro_section
        )

        content_layout.addStretch()

        scroll.setWidget(
            content
        )

        layout.addWidget(
            scroll
        )


    # ========================================================
    # TOP VALUE CARDS / START-STOP CONTROL
    # ========================================================

    def make_top_value(self, title, value):
        card = QFrame()
        card.setObjectName("TopValueCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("TopValueTitle")

        value_label = QLabel(value)
        value_label.setObjectName("TopValue")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        # Keep a direct reference to the value label.
        card.value_label = value_label
        return card

    def clear_sensor_data(self):
        """Clear all old samples so every START begins a fresh recording."""
        for buffer in (
            self.ir_data,
            self.red_data,
            self.bio_data,
            self.ax_data,
            self.ay_data,
            self.az_data,
            self.gx_data,
            self.gy_data,
            self.gz_data,
            self.ml_bio_buffer,
        ):
            buffer.clear()

        self.ml_counter = 0
        self.seizure_positive_count = 0

        self.hr_section.set_value("--", "BPM")
        self.spo2_section.set_value("--", "%")
        self.bio_section.set_value("--", "V")

        self.accel_section.set_xyz(0, 0, 0)
        self.gyro_section.set_xyz(0, 0, 0)

        self.hr_section.info_label.setText("Waiting for samples...")
        self.spo2_section.info_label.setText("Waiting for samples...")
        self.bio_section.info_label.setText("Waiting for samples...")
        self.accel_section.info_label.setText("Waiting for samples...")
        self.gyro_section.info_label.setText("Waiting for samples...")

        if self.calibration_complete:
            self.seizure_status.status.setText("SENSOR FUSION READY")
            self.seizure_status.probability.setText("Fusion Score: --")
            self.seizure_status.info.setText("Press START to begin multimodal seizure screening")
        else:
            self.seizure_status.status.setText("WAITING FOR SENSOR CALIBRATION")
            self.seizure_status.probability.setText("Fusion Score: --")
            self.seizure_status.info.setText("Calibrate HR + SpO2 + BioAmp before START")

        self.fall_alert_timer.stop()
        self.fall_alert_active = False
        self.alert_banner.setText("✓  NO ACTIVE ALERT")
        self.alert_banner.setObjectName("AlertBanner")
        self.alert_banner.style().unpolish(self.alert_banner)
        self.alert_banner.style().polish(self.alert_banner)

        # Do not clear the top live sensor values here.
        # They are continuously displayed even while acquisition is stopped.

        self.latest_hr = None
        self.latest_spo2 = None
        self.latest_bio = None
        self.latest_accel = (None, None, None)
        self.latest_gyro = (None, None, None)

        # Immediately clear visible plots too.
        self.hr_section.line.clear()
        self.spo2_section.line.clear()
        self.bio_section.line.clear()
        self.accel_section.line_x.clear()
        self.accel_section.line_y.clear()
        self.accel_section.line_z.clear()
        self.gyro_section.line_x.clear()
        self.gyro_section.line_y.clear()
        self.gyro_section.line_z.clear()

    def start_calibration(self):
        """Collect a short resting baseline for HR, SpO2 and BioAmp."""
        if self.calibration_active:
            return

        self.calibration_active = True
        self.calibration_complete = False
        self.calibration_started_at = time.monotonic()
        self.calibration_hr = []
        self.calibration_spo2 = []
        self.calibration_bio = []

        self.calibrate_button.setText("⏳ CALIBRATING...")
        self.calibrate_button.setEnabled(False)
        self.calibration_label.setText(
            f"🟡 CALIBRATING — stay still and relaxed for {CALIBRATION_DURATION_SEC:.0f} seconds"
        )
        self.seizure_status.status.setText("SENSOR CALIBRATION")
        self.seizure_status.probability.setText("Fusion Score: --")
        self.seizure_status.info.setText(
            "Collecting resting HR + SpO2 + BioAmp baseline..."
        )

    def update_calibration(self, hr, spo2, bio_v):
        """Add valid live samples and finish the calibration automatically."""
        if not self.calibration_active:
            return

        if hr is not None and hr >= 0:
            self.calibration_hr.append(float(hr))
        if spo2 is not None and spo2 >= 0:
            self.calibration_spo2.append(float(spo2))
        if bio_v is not None and np.isfinite(bio_v):
            self.calibration_bio.append(float(bio_v))

        elapsed = time.monotonic() - self.calibration_started_at
        remaining = max(0.0, CALIBRATION_DURATION_SEC - elapsed)
        self.calibration_label.setText(
            f"🟡 CALIBRATING — {remaining:.1f}s remaining • keep the sensor in its normal resting position"
        )

        if elapsed < CALIBRATION_DURATION_SEC:
            return

        # Require enough data to avoid accepting a bad/empty calibration.
        if len(self.calibration_hr) < 5 or len(self.calibration_spo2) < 5 or len(self.calibration_bio) < 20:
            self.calibration_label.setText(
                "🔴 CALIBRATION FAILED — keep finger/contact stable and press CALIBRATE again"
            )
            self.calibration_active = False
            self.calibrate_button.setEnabled(True)
            self.calibrate_button.setText("⚙  CALIBRATE SENSORS")
            return

        self.hr_baseline = float(np.mean(self.calibration_hr))
        self.hr_baseline_std = float(max(np.std(self.calibration_hr), 3.0))
        self.spo2_baseline = float(np.mean(self.calibration_spo2))
        self.spo2_baseline_std = float(max(np.std(self.calibration_spo2), 0.5))
        self.bio_baseline = float(np.mean(self.calibration_bio))
        self.bio_baseline_std = float(max(np.std(self.calibration_bio), 1e-4))

        self.calibration_active = False
        self.calibration_complete = True
        self.calibrate_button.setEnabled(True)
        self.calibrate_button.setText("✓  RE-CALIBRATE")
        self.calibration_label.setText(
            f"🟢 CALIBRATED — HR {self.hr_baseline:.0f}±{self.hr_baseline_std:.1f} BPM • "
            f"SpO2 {self.spo2_baseline:.1f}±{self.spo2_baseline_std:.1f}% • "
            f"BioAmp {self.bio_baseline:.3f}±{self.bio_baseline_std:.3f} V"
        )
        self.seizure_status.status.setText("SENSOR FUSION READY")
        self.seizure_status.probability.setText("Fusion Score: --")
        self.seizure_status.info.setText(
            "BioAmp + HR + SpO2 baseline calibrated"
        )

    def sensor_fusion_scores(self, bio_probability, hr, spo2):
        """Convert the three calibrated sensor streams into normalized support scores."""
        scores = []
        weights = []
        details = []

        # BioAmp is the primary signal. Its ML probability is the EEG component.
        if bio_probability is not None:
            scores.append(float(np.clip(bio_probability, 0.0, 1.0)))
            weights.append(FUSION_BIO_WEIGHT)
            details.append(f"BioAmp {bio_probability * 100:.0f}%")

        # HR: measure deviation from the person's calibrated resting baseline.
        if self.hr_baseline is not None and hr is not None and hr >= 0:
            hr_z = abs(float(hr) - self.hr_baseline) / self.hr_baseline_std
            hr_score = float(np.clip(hr_z / 3.0, 0.0, 1.0))
            scores.append(hr_score)
            weights.append(FUSION_HR_WEIGHT)
            details.append(f"HR support {hr_score * 100:.0f}%")

        # SpO2: only a drop below the calibrated baseline is treated as support.
        if self.spo2_baseline is not None and spo2 is not None and spo2 >= 0:
            spo2_drop = max(0.0, self.spo2_baseline - float(spo2))
            spo2_z = spo2_drop / self.spo2_baseline_std
            spo2_score = float(np.clip(spo2_z / 3.0, 0.0, 1.0))
            scores.append(spo2_score)
            weights.append(FUSION_SPO2_WEIGHT)
            details.append(f"SpO2 support {spo2_score * 100:.0f}%")

        if not scores:
            return 0.0, "No valid sensors"

        fusion_score = float(np.average(scores, weights=weights))
        return fusion_score, " • ".join(details)

    def toggle_acquisition(self):
        if not self.acquisition_active:
            # Start a completely fresh acquisition.
            self.clear_sensor_data()
            self.acquisition_active = True

            self.start_button.setText("■  STOP")
            self.start_button.setObjectName("StopButton")
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)

            self.acquisition_label.setText("● RECORDING • FALL MONITORING ON")
            if self.calibration_complete:
                self.seizure_status.status.setText("SENSOR FUSION: COLLECTING")
                self.seizure_status.probability.setText("Fusion Score: --")
                self.seizure_status.info.setText("Live BioAmp + HR + SpO2 fusion active • collecting samples...")
            else:
                self.seizure_status.status.setText("WAITING FOR SENSOR CALIBRATION")
                self.seizure_status.probability.setText("Fusion Score: --")
                self.seizure_status.info.setText("Calibrate HR + SpO2 + BioAmp before live seizure screening")

        else:
            # Stop processing incoming samples.
            self.acquisition_active = False

            self.start_button.setText("▶  START")
            self.start_button.setObjectName("StartButton")
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)

            self.acquisition_label.setText("● PAUSED — VITALS + FALL MONITORING ON")
            self.seizure_status.status.setText("SENSOR FUSION OFF — SENSOR STOPPED")
            self.seizure_status.probability.setText("Fusion Score: NOT RUNNING")
            self.seizure_status.info.setText("Fall monitoring and live vitals remain active")

    # ========================================================
    # STYLE
    # ========================================================

    def apply_style(self):

        self.setStyleSheet(
            """
            QWidget {
                background-color: #0F1318;
                color: #F4F6F8;
                font-family: Segoe UI;
            }

            #Header {
                background-color: #151A20;
                border-bottom: 1px solid #242A31;
            }

            #MainTitle {
                font-size: 28px;
                font-weight: 700;
                color: #F7F9FB;
            }

            #Subtitle {
                font-size: 13px;
                color: #98A2AD;
            }

            #Status {
                font-size: 16px;
                font-weight: 700;
                color: #EAF2F8;
            }

            #ControlPanel {
                background-color: #171C22;
                border: 1px solid #242A31;
                border-radius: 18px;
            }

            #AcquisitionLabel {
                font-size: 16px;
                font-weight: 700;
                color: #AEB7C1;
                padding-left: 4px;
            }

            #StartButton, #StopButton {
                border: none;
                border-radius: 10px;
                padding: 10px 22px;
                font-size: 16px;
                font-weight: 800;
                color: white;
            }

            #StartButton {
                background-color: #2E7D32;
            }

            #StartButton:hover {
                background-color: #388E3C;
            }

            #StopButton {
                background-color: #C62828;
            }

            #StopButton:hover {
                background-color: #D32F2F;
            }

            #TopValueCard {
                background-color: #0F141A;
                border: 1px solid #252D36;
                border-radius: 10px;
            }

            #TopValueTitle {
                font-size: 12px;
                font-weight: 700;
                color: #8E99A5;
            }

            #TopValue {
                font-size: 18px;
                font-weight: 800;
                color: #F4F6F8;
            }

            #AlertBanner {
                background-color: #12351E;
                border: 1px solid #2E7D32;
                border-radius: 10px;
                color: #B9F6CA;
                font-size: 15px;
                font-weight: 800;
                padding: 7px;
            }

            #CalibrateButton {
                background-color: #455A64;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
                font-weight: 700;
            }

            #CalibrateButton:hover {
                background-color: #546E7A;
            }

            #CalibrationLabel {
                color: #AEB7C1;
                font-size: 13px;
                padding: 4px;
            }

            #AlertBannerHigh {
                background-color: #4A1717;
                border: 1px solid #EF5350;
                border-radius: 10px;
                color: #FFCDD2;
                font-size: 15px;
                font-weight: 800;
                padding: 7px;
            }

            #Toast {
                background-color: #7A1F1F;
                border: 2px solid #EF5350;
                border-radius: 12px;
                color: white;
                font-size: 18px;
                font-weight: 800;
                padding: 14px 28px;
            }

            #FileAnalysisPanel {
                background-color: #171C22;
                border: 1px solid #242A31;
                border-radius: 18px;
            }

            #FileAnalysisTitle {
                font-size: 20px;
                font-weight: 800;
                color: #F8FAFC;
            }

            #FilePathLabel {
                font-size: 13px;
                color: #AEB7C1;
                padding: 8px;
            }

            #ChooseFileButton, #AnalyzeFileButton {
                border: none;
                border-radius: 9px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 800;
                color: white;
            }

            #ChooseFileButton {
                background-color: #37474F;
            }

            #ChooseFileButton:hover {
                background-color: #455A64;
            }

            #AnalyzeFileButton {
                background-color: #1565C0;
            }

            #AnalyzeFileButton:hover {
                background-color: #1976D2;
            }

            #AnalyzeFileButton:disabled {
                background-color: #303840;
                color: #77818C;
            }

            #FileResult {
                background-color: #0F141A;
                border: 1px solid #252D36;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
                font-weight: 800;
                color: #F4F6F8;
            }

            #SensorSection {
                background-color: #171C22;
                border-radius: 18px;
            }

            #SensorTitle {
                font-size: 21px;
                font-weight: 700;
                color: #F8FAFC;
            }

            #MainValue {
                font-size: 58px;
                font-weight: 700;
                color: white;
            }

            #MainUnit {
                font-size: 17px;
                color: #AEB7C1;
            }

            #XYZLarge {
                font-size: 24px;
                font-weight: 600;
                color: #EDF2F7;
            }

            #InfoText {
                font-size: 14px;
                color: #98A2AD;
            }

            #SeizureStatus {
                background-color: #171C22;
                border-radius: 18px;
                border: 1px solid #242A31;
            }

            #SeizureTitle {
                font-size: 21px;
                font-weight: 700;
                color: #F8FAFC;
            }

            #SeizureState {
                font-size: 30px;
                font-weight: 800;
                color: #F8FAFC;
                padding: 8px;
            }

            #SeizureProbability {
                font-size: 18px;
                font-weight: 600;
                color: #C9D2DB;
            }

            #SeizureInfo {
                font-size: 13px;
                color: #98A2AD;
            }

            #SeizureDisclaimer {
                font-size: 11px;
                color: #707B87;
                padding-top: 2px;
            }

            QScrollBar:vertical {
                background: #11161C;
                width: 12px;
                margin: 4px;
            }

            QScrollBar::handle:vertical {
                background: #45515E;
                min-height: 50px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background: #607080;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )


    # ========================================================
    # CONNECTION
    # ========================================================

    def update_status(
        self,
        status
    ):

        self.status_label.setText(
            f"● {status}"
        )


    # ========================================================
    # UPLOADED EEG FILE ANALYSIS
    # ========================================================

    def choose_eeg_file(self):
        if mne is None:
            self.file_result.setText(
                "MNE is not installed. Install it with: pip install mne"
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG recording",
            "",
            "EDF EEG files (*.edf);;All files (*)"
        )

        if not file_path:
            return

        self.selected_eeg_file = file_path
        self.file_path_label.setText(file_path)
        self.analyze_file_button.setEnabled(True)
        self.file_result.setText(
            "File selected. Click ANALYZE EEG."
        )

    def analyze_eeg_file(self):
        """Analyze an uploaded CHB-MIT EDF using its annotation only.

        IMPORTANT: this function NEVER runs the live ML classifier.
        Uploaded EEG is displayed and checked against the CHB-MIT
        seizure annotation. The ML model is reserved for live ESP32
        BioAmp acquisition after START is pressed.
        """
        if self.selected_eeg_file is None:
            return

        if mne is None:
            self.file_result.setText(
                "MNE is not installed. Install it with:\n"
                "pip install mne"
            )
            return

        self.choose_file_button.setEnabled(False)
        self.analyze_file_button.setEnabled(False)
        self.file_progress.show()
        self.file_progress.setValue(5)

        try:
            edf_path = Path(self.selected_eeg_file)
            annotation_path = find_annotation_file(edf_path)

            if annotation_path is None:
                self.file_result.setText(
                    "⚠ NO SEIZURE ANNOTATION FOUND\n\n"
                    f"EDF: {edf_path.name}\n"
                    "This file cannot be classified as seizure/no-seizure "
                    "from the CHB-MIT annotation because its annotation "
                    "file is missing.\n\n"
                    "ML STATUS: OFF — uploaded-file analysis does not run "
                    "the live sensor ML model."
                )
                self.eeg_line.clear()
                self.file_progress.setValue(100)
                return

            intervals = parse_seizure_file(annotation_path)
            self.file_progress.setValue(20)

            raw = mne.io.read_raw_edf(
                str(edf_path), preload=True, verbose="ERROR"
            )
            channel_name = choose_eeg_channel(raw)
            signal = raw.get_data(picks=[channel_name])[0].astype(np.float32)
            original_sfreq = float(raw.info["sfreq"])

            if abs(original_sfreq - 256.0) > 0.01:
                from scipy.signal import resample
                target_len = max(1, int(len(signal) * 256.0 / original_sfreq))
                signal = resample(signal, target_len).astype(np.float32)
                sfreq = 256.0
            else:
                sfreq = original_sfreq

            signal = signal - np.median(signal)
            scale = float(np.std(signal))
            if scale > 1e-8:
                signal = signal / scale

            # Plot a readable representation without sending every sample
            # to the GUI when a long EDF contains hundreds of thousands of points.
            max_points = 12000
            if len(signal) > max_points:
                idx = np.linspace(0, len(signal) - 1, max_points).astype(int)
                plot_signal = signal[idx]
                plot_time = idx / sfreq
            else:
                plot_signal = signal
                plot_time = np.arange(len(signal), dtype=np.float32) / sfreq

            self.eeg_line.setData(plot_time, plot_signal)
            self.eeg_plot.setTitle(
                f"{edf_path.name}  •  Channel: {channel_name}",
                color="white"
            )

            # Remove seizure regions created by the previous analysis.
            # Do not iterate over self.eeg_plot.items here: in some
            # pyqtgraph versions that name resolves to a method, which
            # causes: "builtin_function_or_method object is not iterable".
            old_regions = getattr(self, "eeg_regions", [])
            for region in old_regions:
                try:
                    self.eeg_plot.removeItem(region)
                except Exception:
                    pass
            self.eeg_regions = []

            self.file_progress.setValue(60)

            if intervals:
                for start_sec, end_sec in intervals:
                    region = pg.LinearRegionItem(
                        values=[start_sec, end_sec],
                        movable=False,
                        brush=pg.mkBrush(220, 60, 60, 60),
                        pen=pg.mkPen(220, 80, 80, width=1.5),
                    )
                    self.eeg_plot.addItem(region)
                    self.eeg_regions.append(region)

                details = "\n".join(
                    f"Seizure {i}: {a:.0f}–{b:.0f} sec  "
                    f"(duration {b-a:.0f} sec)"
                    for i, (a, b) in enumerate(intervals, start=1)
                )

                reason = (
                    "The CHB-MIT annotation marks the interval shown below "
                    "as a seizure event. The highlighted part of the EEG graph "
                    "corresponds to that annotated interval."
                )
                observation = (
                    "The EEG waveform shown in the highlighted interval is the "
                    "recorded brain-signal segment associated with the annotated event. "
                    "This result does not determine the medical cause of the seizure."
                )

                self.file_result.setText(
                    "⚠ SEIZURE DETECTED\n\n"
                    f"File: {edf_path.name}\n"
                    f"Channel: {channel_name}\n"
                    f"Sampling rate: {sfreq:.0f} Hz\n\n"
                    f"{details}\n\n"
                    "WHY THIS RESULT?\n"
                    f"{reason}\n\n"
                    "WHAT THE GRAPH SHOWS\n"
                    f"{observation}\n\n"
                    "CONCLUSION\n"
                    "A seizure event is present in this CHB-MIT recording according to its annotation."
                )

                self.seizure_status.status.setText("FILE: SEIZURE ANNOTATED")
                self.seizure_status.probability.setText("Detection Source: CHB-MIT Annotation")
                self.seizure_status.info.setText(
                    "Reason: seizure interval is marked in the dataset annotation"
                )
            else:
                self.file_result.setText(
                    "✓ NO SEIZURE DETECTED IN ANNOTATION\n\n"
                    f"File: {edf_path.name}\n"
                    f"Channel: {channel_name}\n"
                    f"Sampling rate: {sfreq:.0f} Hz\n\n"
                    "WHY THIS RESULT?\n"
                    "The CHB-MIT annotation contains no seizure interval for this recording.\n\n"
                    "WHAT THE GRAPH SHOWS\n"
                    "The displayed EEG is the recorded signal for the selected channel, "
                    "with no seizure interval marked by the dataset annotation.\n\n"
                    "CONCLUSION\n"
                    "No annotated seizure event was found in this recording. This does not prove that an unknown, unannotated EEG is medically seizure-free."
                )
                self.seizure_status.status.setText("FILE: NO SEIZURE ANNOTATED")
                self.seizure_status.probability.setText("Detection Source: CHB-MIT Annotation")
                self.seizure_status.info.setText(
                    "Reason: no seizure interval is marked in the dataset annotation"
                )

            self.file_progress.setValue(100)

        except Exception as exc:
            self.file_result.setText(
                "Could not analyze EEG file:\n"
                f"{exc}"
            )
            self.file_progress.setValue(0)

        finally:
            self.choose_file_button.setEnabled(True)
            self.analyze_file_button.setEnabled(
                self.selected_eeg_file is not None
            )

    # ========================================================
    # ALERT SYSTEM
    # ========================================================

    def show_toast(self, message, duration_ms=HR_ALERT_COOLDOWN_MS):
        """Show a short, non-blocking alert on top of the dashboard."""
        self.toast.setText(message)
        self.toast.adjustSize()

        # Keep it near the top-center of the application.
        x = max(0, (self.width() - self.toast.width()) // 2)
        y = 105
        self.toast.move(x, y)

        self.toast.show()
        self.toast.raise_()
        self.toast_timer.start(duration_ms)

    def update_hr_alert(self, hr):
        """
        Heart-rate alerts are independent of START/STOP.
        This means live HR can be monitored even when EEG/BioAmp
        acquisition and seizure ML are not running.
        """
        if hr is None or hr < 0:
            return

        now_ms = int(time.monotonic() * 1000)

        if hr >= HIGH_HR_THRESHOLD:
            message = (
                f"⚠ HEART RATE HIGH — {hr:.0f} BPM "
                f"(limit {HIGH_HR_THRESHOLD:.0f})"
            )

            self.alert_banner.setText(message)
            self.alert_banner.setObjectName("AlertBannerHigh")

            # Avoid repeatedly flashing a toast on every serial sample.
            if (
                not self.hr_alert_active
                or now_ms - self.last_hr_alert_ms >= HR_ALERT_COOLDOWN_MS
            ):
                self.show_toast(message)
                self.last_hr_alert_ms = now_ms

            self.hr_alert_active = True

        elif hr <= LOW_HR_THRESHOLD:
            message = (
                f"⚠ HEART RATE LOW — {hr:.0f} BPM "
                f"(limit {LOW_HR_THRESHOLD:.0f})"
            )

            self.alert_banner.setText(message)
            self.alert_banner.setObjectName("AlertBannerHigh")

            if (
                not self.hr_alert_active
                or now_ms - self.last_hr_alert_ms >= HR_ALERT_COOLDOWN_MS
            ):
                self.show_toast(message)
                self.last_hr_alert_ms = now_ms

            self.hr_alert_active = True

        else:
            # Back inside the configured range.
            if self.hr_alert_active:
                self.alert_banner.setText("✓  HEART RATE BACK IN RANGE")
                self.alert_banner.setObjectName("AlertBanner")

                # Re-polish because the object name controls the CSS.
                self.alert_banner.style().unpolish(self.alert_banner)
                self.alert_banner.style().polish(self.alert_banner)

            else:
                self.alert_banner.setText("✓  NO ACTIVE ALERT")

            self.hr_alert_active = False

        self.alert_banner.style().unpolish(self.alert_banner)
        self.alert_banner.style().polish(self.alert_banner)

    # ========================================================
    # FALL DETECTION
    # ========================================================

    def update_fall_detection(self, gx, gy, gz):
        """
        Detect a suspected fall from a sudden high angular-speed
        event measured by the MPU6050 gyroscope.

        The three gyro axes are combined into one magnitude:
        sqrt(gx^2 + gy^2 + gz^2). A few consecutive samples above
        the threshold are required to reduce false alarms.
        """
        # Fall monitoring is intentionally independent of START/STOP.
        # The ESP32 serial reader remains active, so a fall can be
        # detected even while the main acquisition is stopped.
        gyro_magnitude = float(np.sqrt(gx * gx + gy * gy + gz * gz))

        # One sample crossing the threshold is enough.
        # The detector is edge-triggered: while gyro remains above the
        # threshold, it will not repeatedly fire. It becomes armed again
        # only after the gyro falls back below the threshold.
        if gyro_magnitude < FALL_GYRO_THRESHOLD:
            self.fall_candidate_count = 0
            self.fall_armed = True
            return

        if not self.fall_armed:
            return

        self.fall_candidate_count = FALL_CONFIRM_SAMPLES
        self.fall_armed = False

        now_ms = int(time.monotonic() * 1000)

        message = (
            f"⚠ SUSPECTED FALL DETECTED — Gyro {gyro_magnitude:.0f} °/s "
            f"(threshold {FALL_GYRO_THRESHOLD:.0f} °/s)"
        )

        self.alert_banner.setText(message)
        self.alert_banner.setObjectName("AlertBannerHigh")
        self.alert_banner.style().unpolish(self.alert_banner)
        self.alert_banner.style().polish(self.alert_banner)

        # Fall alert is intentionally temporary.
        # Keep it visible for 5 seconds, then clear it automatically.
        self.show_toast(message, FALL_ALERT_DISPLAY_MS)
        self.fall_alert_timer.start(FALL_ALERT_DISPLAY_MS)

        self.last_fall_alert_ms = now_ms
        self.fall_alert_active = True
        self.fall_candidate_count = 0

    def clear_fall_alert(self):
        """Remove the fall alert after its 5-second display period."""
        self.fall_alert_active = False

        # If HR is currently outside its configured range, keep the HR
        # alert visible instead of replacing it with NO ACTIVE ALERT.
        if self.latest_hr is not None and (
            self.latest_hr >= HIGH_HR_THRESHOLD
            or self.latest_hr <= LOW_HR_THRESHOLD
        ):
            self.update_hr_alert(self.latest_hr)
            return

        self.alert_banner.setText("✓  NO ACTIVE ALERT")
        self.alert_banner.setObjectName("AlertBanner")
        self.alert_banner.style().unpolish(self.alert_banner)
        self.alert_banner.style().polish(self.alert_banner)


    # ========================================================
    # DATA
    # ========================================================

    def process_data(
        self,
        data
    ):

        (
            millis,
            hr,
            spo2,
            ir,
            red,
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            bio_raw,
            bio_v
        ) = data

        # ====================================================
        # HR
        # ====================================================

        if hr >= 0:

            # Live HR display and alert work even before START.
            self.hr_section.set_value(
                f"{hr:.0f}",
                "BPM"
            )
            self.latest_hr = hr
            self.top_hr.value_label.setText(f"{hr:.0f} BPM")
            self.update_hr_alert(hr)

        else:

            self.hr_section.set_value(
                "--",
                "BPM"
            )
            self.top_hr.value_label.setText("-- BPM")

        finger_text = (
            "Finger detected"
            if ir >= FINGER_THRESHOLD
            else "Finger not detected"
        )

        self.hr_section.info_label.setText(
            f"{finger_text}   •   IR: {ir:,}"
        )

        # ====================================================
        # SPO2
        # ====================================================

        if spo2 >= 0:

            self.spo2_section.set_value(
                str(spo2),
                "%"
            )
            self.latest_spo2 = spo2
            self.top_spo2.value_label.setText(f"{spo2} %")

        else:

            self.spo2_section.set_value(
                "--",
                "%"
            )

        self.spo2_section.info_label.setText(
            f"RED: {red:,}"
        )

        # ====================================================
        # BIOAMP
        # ====================================================

        self.bio_section.set_value(
            f"{bio_v:.3f}",
            "V"
        )
        self.latest_bio = bio_v
        self.top_bio.value_label.setText(f"{bio_v:.3f} V")

        self.bio_section.info_label.setText(
            f"Raw ADC: {bio_raw}"
        )

        # ====================================================
        # ACCEL
        # ====================================================

        self.accel_section.set_xyz(
            ax,
            ay,
            az
        )
        self.latest_accel = (ax, ay, az)
        self.top_accel.value_label.setText(
            f"X {ax:+.2f}  Y {ay:+.2f}  Z {az:+.2f} g"
        )

        self.accel_section.info_label.setText(
            "X = Blue   •   Y = Orange   •   Z = Green"
        )

        # ====================================================
        # GYRO
        # ====================================================

        self.gyro_section.set_xyz(
            gx,
            gy,
            gz
        )
        self.latest_gyro = (gx, gy, gz)
        self.top_gyro.value_label.setText(
            f"X {gx:+.1f}  Y {gy:+.1f}  Z {gz:+.1f} °/s"
        )

        self.gyro_section.info_label.setText(
            "X = Blue   •   Y = Orange   •   Z = Green"
        )

        # Sensor calibration is independent of START/STOP.
        self.update_calibration(hr, spo2, bio_v)

        # FALL DETECTION IS ALWAYS ACTIVE.
        # It must continue working even when START/STOP is OFF.
        self.update_fall_detection(gx, gy, gz)

        # ====================================================
        # ACQUISITION GATE
        # ====================================================
        # Sensor values above are live display values.
        # From this point onward, data is recorded/processed ONLY
        # when START has been pressed.
        if not self.acquisition_active:
            return

        # ====================================================
        # BUFFER
        # ====================================================

        self.ir_data.append(
            ir
        )

        self.red_data.append(
            red
        )

        self.bio_data.append(
            bio_v
        )

        # ====================================================
        # ML SEIZURE DETECTION
        # ====================================================

        self.run_sensor_fusion_detection(
            bio_v,
            hr,
            spo2
        )

        self.ax_data.append(
            ax
        )

        self.ay_data.append(
            ay
        )

        self.az_data.append(
            az
        )

        self.gx_data.append(
            gx
        )

        self.gy_data.append(
            gy
        )

        self.gz_data.append(
            gz
        )



    # ========================================================
    # ML SEIZURE DETECTION
    # ========================================================

    def run_sensor_fusion_detection(self, bio_v, hr, spo2):
        """
        Live multimodal seizure screening using calibrated BioAmp + HR + SpO2.

        BioAmp remains the primary signal. HR and SpO2 are corroborating
        physiological signals; they do not independently diagnose a seizure.
        """
        if not self.calibration_complete:
            self.seizure_status.status.setText("WAITING FOR SENSOR CALIBRATION")
            self.seizure_status.probability.setText("Fusion Score: --")
            self.seizure_status.info.setText(
                "Press CALIBRATE SENSORS before starting live sensor-fusion detection"
            )
            return

        self.ml_bio_buffer.append(float(bio_v))
        self.ml_counter += 1

        if len(self.ml_bio_buffer) < ML_WINDOW_SAMPLES:
            remaining = ML_WINDOW_SAMPLES - len(self.ml_bio_buffer)
            self.seizure_status.status.setText("SENSOR FUSION: COLLECTING")
            self.seizure_status.probability.setText("Fusion Score: --")
            self.seizure_status.info.setText(
                f"Collecting multimodal window... {remaining} BioAmp samples remaining"
            )
            return

        if self.ml_counter < ML_STEP_SAMPLES:
            return

        self.ml_counter = 0

        bio_probability, state = self.ml_model.predict(list(self.ml_bio_buffer))
        fusion_score, details = self.sensor_fusion_scores(
            bio_probability, hr, spo2
        )

        # Require strong BioAmp evidence as well as a high multimodal score.
        # This prevents a change in HR/SpO2 by itself from being labelled a seizure.
        positive = (
            bio_probability >= FUSION_BIO_MIN_PROBABILITY
            and fusion_score >= FUSION_THRESHOLD
        )

        if positive:
            self.seizure_positive_count += 1
        else:
            self.seizure_positive_count = 0

        if self.seizure_positive_count >= ML_CONFIRM_WINDOWS:
            self.seizure_status.status.setText("⚠ MULTIMODAL SEIZURE-LIKE PATTERN")
            self.seizure_status.probability.setText(
                f"Fusion Score: {fusion_score * 100:.1f}% • BioAmp: {bio_probability * 100:.1f}%"
            )
            self.seizure_status.info.setText(
                f"Confirmed across {ML_CONFIRM_WINDOWS} windows • {details}"
            )

        elif positive:
            self.seizure_status.status.setText("POSSIBLE MULTIMODAL EVENT")
            self.seizure_status.probability.setText(
                f"Fusion Score: {fusion_score * 100:.1f}% • BioAmp: {bio_probability * 100:.1f}%"
            )
            self.seizure_status.info.setText(
                f"Confirmation: {self.seizure_positive_count}/{ML_CONFIRM_WINDOWS} windows • {details}"
            )

        else:
            self.seizure_status.status.setText("NORMAL / NO FUSED EVENT")
            self.seizure_status.probability.setText(
                f"Fusion Score: {fusion_score * 100:.1f}% • BioAmp: {bio_probability * 100:.1f}%"
            )
            self.seizure_status.info.setText(
                f"No high-confidence multimodal event • {details}"
            )


    # ========================================================
    # GRAPH REFRESH
    # ========================================================

    def refresh_graphs(self):

        if len(self.ir_data) > 2:

            self.hr_section.line.setData(
                list(
                    self.ir_data
                )
            )

        if len(self.red_data) > 2:

            self.spo2_section.line.setData(
                list(
                    self.red_data
                )
            )

        if len(self.bio_data) > 2:

            self.bio_section.line.setData(
                list(
                    self.bio_data
                )
            )

        if len(self.ax_data) > 2:

            self.accel_section.line_x.setData(
                list(
                    self.ax_data
                )
            )

            self.accel_section.line_y.setData(
                list(
                    self.ay_data
                )
            )

            self.accel_section.line_z.setData(
                list(
                    self.az_data
                )
            )

        if len(self.gx_data) > 2:

            self.gyro_section.line_x.setData(
                list(
                    self.gx_data
                )
            )

            self.gyro_section.line_y.setData(
                list(
                    self.gy_data
                )
            )

            self.gyro_section.line_z.setData(
                list(
                    self.gz_data
                )
            )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.acquisition_active = False

        try:

            self.reader.stop()

        except:
            pass

        event.accept()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    pg.setConfigOptions(
        antialias=False
    )

    dashboard = Dashboard()

    dashboard.show()

    sys.exit(
        app.exec()
    )

