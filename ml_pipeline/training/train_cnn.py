from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
MEL_CACHE_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "mel_specs.npz"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"
CRITICAL_CLASSES = ["gunshot", "glass_breaking", "panic_scream", "aggression", "person_asking_for_help"]

TARGET_SR = 22050
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 66150
N_MELS = 64
N_FFT = 512
HOP_LEN = 256
TIME_STEPS = 128


def create_mel_filterbank(sr: int = TARGET_SR, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * float(np.log10(1.0 + f / 700.0))

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


MEL_FILTERBANK = create_mel_filterbank()


def resolve_audio_file(row: Dict[str, str]) -> Optional[Path]:
    filename = row.get("filename", "")
    cls_name = row.get("class", "")
    final_split = row.get("final_split") or "train"
    source_split = row.get("source_split") or "train"
    is_augmented = row.get("original_or_augmented") == "augmented"

    if is_augmented:
        p = REPO_ROOT / "audio_dataset" / "augmented" / cls_name / filename
        if p.exists():
            return p

    p = REPO_ROOT / "audio_dataset" / "preprocessed" / final_split / cls_name / filename
    if p.exists():
        return p

    p = REPO_ROOT / "audio_dataset" / final_split / cls_name / filename
    if p.exists():
        return p

    p = REPO_ROOT / "audio_dataset" / "raw" / source_split / cls_name / filename
    if p.exists():
        return p

    return None


def extract_mel_spectrogram(filepath: Path) -> Optional[np.ndarray]:
    try:
        y, sr = sf.read(str(filepath), frames=TARGET_SAMPLES, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if len(y) == 0:
            return None
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        else:
            y = y[:TARGET_SAMPLES]

        _, _, Zxx = signal.stft(y, fs=TARGET_SR, nperseg=N_FFT, noverlap=N_FFT - HOP_LEN)
        melspec = np.dot(MEL_FILTERBANK, np.abs(Zxx))
        melspec_db = 10.0 * np.log10(np.maximum(melspec, 1e-10))

        if melspec_db.shape[1] >= TIME_STEPS * 2:
            img = melspec_db[:, : TIME_STEPS * 2].reshape(N_MELS, TIME_STEPS, 2).mean(axis=2)
        elif melspec_db.shape[1] >= TIME_STEPS:
            img = melspec_db[:, :TIME_STEPS]
        else:
            img = np.pad(melspec_db, ((0, 0), (0, TIME_STEPS - melspec_db.shape[1])))

        std = float(np.std(img))
        img = (img - float(np.mean(img))) / (std if std > 1e-6 else 1.0)
        return img[np.newaxis, :, :].astype(np.float32)
    except Exception:
        return None


def load_mel_dataset(cache_path: Path = MEL_CACHE_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Mel-spectrogram cache not found at {cache_path}")

    print(f"Loading cached mel-spectrogram representations from {cache_path.relative_to(REPO_ROOT)} ...")
    data = np.load(cache_path)
    X_tr = data["X_train"].astype(np.float32)
    y_tr = data["y_train"]
    X_va = data["X_val"].astype(np.float32)
    y_val = data["y_val"]
    classes = data["classes"].tolist()
    print(f"Loaded {len(X_tr)} train and {len(X_va)} val mel-spectrograms (1x{N_MELS}x{TIME_STEPS}) across {len(classes)} classes.")
    return X_tr, y_tr, X_va, y_val, classes


def balance_subsample(X: np.ndarray, y: np.ndarray, max_samples_per_class: int = 600, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_seed)
    unique_classes = np.unique(y)
    selected_indices = []

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > max_samples_per_class:
            chosen = rng.choice(cls_idx, size=max_samples_per_class, replace=False)
        else:
            chosen = cls_idx
        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    idx_arr = np.array(selected_indices)
    return X[idx_arr], y[idx_arr]


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    weights = np.clip(weights, 0.3, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_single_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(y_b)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y_b).sum().item())
        total += len(y_b)

    return total_loss / total, correct / total


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            total_loss += float(loss.item()) * len(y_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y_b.cpu().numpy().tolist())
            total += len(y_b)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = float(accuracy_score(y_true, y_pred))
    return total_loss / total, acc, y_true, y_pred


def train_cnn(fast_tune: bool = False, max_samples_per_class: int = 600, epochs: int = 4) -> Dict[str, Any]:
    X_tr_full, y_tr_str_full, X_va, y_va_str, classes = load_mel_dataset()

    print(f"Balancing training split (max {max_samples_per_class} samples per class) ...")
    X_tr, y_tr_str = balance_subsample(X_tr_full, y_tr_str_full, max_samples_per_class=max_samples_per_class)

    le = LabelEncoder()
    le.fit(classes)
    y_tr = le.transform(y_tr_str)
    y_va = le.transform(y_va_str)
    num_classes = len(classes)

    print(f"Training subset: {len(X_tr)} balanced samples across {num_classes} classes.")
    print(f"Validation set : {len(X_va)} full samples across {num_classes} classes.")

    class_weights = compute_class_weights(y_tr, num_classes)
    device = torch.device("cpu")

    if fast_tune:
        grid = [
            {"lr": 0.002, "batch_size": 128, "dropout": 0.2, "epochs": epochs},
            {"lr": 0.001, "batch_size": 128, "dropout": 0.25, "epochs": epochs},
        ]
    else:
        grid = [
            {"lr": 0.002, "batch_size": 128, "dropout": 0.2, "epochs": epochs},
            {"lr": 0.001, "batch_size": 128, "dropout": 0.25, "epochs": epochs},
            {"lr": 0.001, "batch_size": 64, "dropout": 0.2, "epochs": epochs},
        ]

    val_dataset = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"\nEvaluating {len(grid)} CNN configurations on validation split ...")
    history: List[Dict[str, Any]] = []
    best_f1, best_params, best_model_state = -1.0, grid[0], None

    for i, p in enumerate(grid, 1):
        t0 = time.time()
        train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
        train_loader = DataLoader(train_dataset, batch_size=p["batch_size"], shuffle=True)

        model = AudioCNN(num_classes=num_classes, dropout=p["dropout"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=1e-4)

        ep_count = p["epochs"]
        best_trial_f1 = -1.0
        best_trial_acc = 0.0

        for ep in range(1, ep_count + 1):
            ep_t0 = time.time()
            tr_loss, tr_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, y_true, y_pred = evaluate_model(model, val_loader, criterion, device)
            va_f1 = float(f1_score(y_true, y_pred, average="macro"))
            ep_time = time.time() - ep_t0

            print(
                f"    [Trial {i} Epoch {ep}/{ep_count}] "
                f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.1f}% | "
                f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc*100:.2f}%, Val F1: {va_f1:.4f} ({ep_time:.1f}s)",
                flush=True,
            )

            if va_f1 > best_trial_f1:
                best_trial_f1 = va_f1
                best_trial_acc = va_acc
                if va_f1 > best_f1:
                    best_f1 = va_f1
                    best_params = p
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        history.append({
            "trial": i,
            "params": p,
            "val_accuracy": round(best_trial_acc, 6),
            "val_macro_f1": round(best_trial_f1, 6),
            "train_time_seconds": round(elapsed, 3),
        })
        print(
            f"  Trial {i}/{len(grid)}: lr={p['lr']}, bs={p['batch_size']}, drop={p['dropout']} "
            f"-> Best Val Acc: {best_trial_acc*100:.2f}%, Best Val F1: {best_trial_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

    print(f"\nOptimal CNN: lr={best_params['lr']}, batch_size={best_params['batch_size']}, dropout={best_params['dropout']} (Val F1: {best_f1:.4f})")

    # Final Model Evaluation
    final_model = AudioCNN(num_classes=num_classes, dropout=best_params["dropout"]).to(device)
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Evaluate train and val
    train_eval_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()), batch_size=128, shuffle=False)
    _, tr_acc, tr_true, tr_pred = evaluate_model(final_model, train_eval_loader, criterion, device)
    _, va_acc, va_true, va_pred = evaluate_model(final_model, val_loader, criterion, device)

    tr_str_true = le.inverse_transform(tr_true)
    tr_str_pred = le.inverse_transform(tr_pred)
    va_str_true = le.inverse_transform(va_true)
    va_str_pred = le.inverse_transform(va_pred)

    tr_f1 = float(f1_score(tr_str_true, tr_str_pred, average="macro"))
    va_f1 = float(f1_score(va_str_true, va_str_pred, average="macro"))

    report = classification_report(va_str_true, va_str_pred, target_names=classes, output_dict=True)
    conf = confusion_matrix(va_str_true, va_str_pred, labels=classes).tolist()

    crit = {}
    print("\nSRS Critical-Class Recalls (SRS §1.7 Target: >= 85%):")
    for cls in CRITICAL_CLASSES:
        rec = float(report[cls]["recall"]) if cls in report else 0.0
        crit[cls] = round(rec, 4)
        status = "PASSED (>= 85%)" if rec >= 0.85 else "FAILED (< 85%)"
        print(f"  SRS Critical Recall [{cls}]: {rec*100:.2f}% -> {status}")

    print(f"\nCNN Final Performance:")
    print(f"  Training Accuracy   : {tr_acc*100:.2f}% | Macro F1: {tr_f1:.4f}")
    print(f"  Validation Accuracy : {va_acc*100:.2f}% | Macro F1: {va_f1:.4f}")

    total_training_time = sum(h["train_time_seconds"] for h in history)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model state and metadata
    model_artifact = {
        "model_state_dict": final_model.state_dict(),
        "model_architecture": {
            "num_classes": num_classes,
            "in_channels": 1,
            "n_mels": N_MELS,
            "target_len": TARGET_SAMPLES,
            "sr": TARGET_SR,
        },
        "classes": classes,
        "best_hyperparameters": best_params,
        "model_version": "cnn_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    mp = RESULTS_DIR / "cnn_model.pt"
    torch.save(model_artifact, mp)
    print(f"\nSaved model artifact to {mp.relative_to(REPO_ROOT)}")

    sp = ML_SERVICE_ARTIFACTS_DIR / "cnn_model.pt"
    shutil.copy2(mp, sp)
    print(f"Exported model copy to {sp.relative_to(REPO_ROOT)}")

    results_payload = {
        "model_name": "Convolutional Neural Network (CNN)",
        "model_version": "cnn_v1.0",
        "best_hyperparameters": best_params,
        "training_time_seconds": round(total_training_time, 3),
        "train_accuracy": round(tr_acc, 6),
        "val_accuracy": round(va_acc, 6),
        "train_macro_f1": round(tr_f1, 6),
        "val_macro_f1": round(va_f1, 6),
        "critical_class_recalls": crit,
        "hyperparameter_search": history,
        "classification_report": report,
        "confusion_matrix": conf,
        "classes": classes,
    }

    rp = RESULTS_DIR / "cnn_results.json"
    with rp.open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"Saved evaluation metrics to {rp.relative_to(REPO_ROOT)}")

    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN directly on mel-spectrogram representations.")
    parser.add_argument("--fast-tune", action="store_true", help="Run 2-candidate fast search")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per candidate trial")
    parser.add_argument("--max-samples-per-class", type=int, default=600, help="Max samples per class for training")
    args = parser.parse_args()
    train_cnn(fast_tune=args.fast_tune, max_samples_per_class=args.max_samples_per_class, epochs=args.epochs)


if __name__ == "__main__":
    main()
