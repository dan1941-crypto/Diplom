"""Training, evaluation (Macro-F1, PR-AUC, FPR), and ONNX export pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig, load_config
from src.features.extractor import FEATURE_NAMES, NUM_FEATURES
from src.features.preprocessor import FlowPreprocessor
from src.ml.dataset import (
    CLASS_NAMES,
    build_autonomous_dataset,
    load_combined_real_datasets,
)
from src.ml.model import FocalLoss, TabularResNet


def _compute_binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute False Positive Rate where class 0 (BENIGN) is negative and >=1 is positive."""
    neg_mask = y_true == 0
    if not np.any(neg_mask):
        return 0.0
    fp = np.sum((y_pred != 0) & neg_mask)
    tn = np.sum((y_pred == 0) & neg_mask)
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def _compute_macro_pr_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Compute macro-averaged Precision-Recall AUC."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    scores: list[float] = []
    for c in range(num_classes):
        if np.sum(y_bin[:, c]) > 0:
            scores.append(float(average_precision_score(y_bin[:, c], probs[:, c])))
    return float(np.mean(scores)) if scores else 0.0


def train_and_export(
    cfg: AppConfig | None = None,
    total_samples: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train TabularResNet on shuffled data, evaluate metrics, and export ONNX + preprocessor."""
    cfg = cfg or load_config()
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    def _emit(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    _emit({"stage": "data", "device": device_name, "message": f"Загрузка датасетов (устройство: {device_name})..."})

    combined = load_combined_real_datasets(cfg.data_dir, seed=cfg.random_seed) if total_samples is None else None
    if combined is not None:
        x, y, class_names, sources = combined
        dataset_source = " + ".join(sources)
    else:
        n_samples = total_samples if total_samples is not None else 12_000
        x, y = build_autonomous_dataset(total_samples=n_samples, seed=cfg.random_seed)
        class_names = CLASS_NAMES
        dataset_source = "autonomous_synthetic"

    num_classes = len(class_names)
    info_msg = f"Device: {device_name} | Источники: {dataset_source} | Всего: {len(y):,} | Классов: {num_classes}"
    print(f"[Train] {info_msg}", flush=True)
    _emit({"stage": "data", "device": device_name, "message": info_msg})

    # Stratified split: train / val / test
    test_val_ratio = cfg.val_split + cfg.test_split
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_val_ratio, stratify=y, random_state=cfg.random_seed
    )
    rel_test = cfg.test_split / test_val_ratio
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=cfg.random_seed
    )

    # Fit preprocessor strictly on shuffled X_train
    preprocessor = FlowPreprocessor(clip_max=cfg.clip_max)
    x_train_s = preprocessor.fit_transform(x_train)
    x_val_s = preprocessor.transform(x_val)
    x_test_s = preprocessor.transform(x_test)

    # Inverse-frequency class weights for Focal Loss (dampened with sqrt, keeping Benign FPR low)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = np.sqrt(counts.sum() / (num_classes * np.maximum(counts, 1.0)))
    weights = weights / weights.mean()
    weights[0] = max(float(weights[0]), 1.25)
    alpha = torch.tensor(weights, dtype=torch.float32, device=device)

    model = TabularResNet(
        in_features=NUM_FEATURES,
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y_train)),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    x_val_t = torch.from_numpy(x_val_s).to(device)
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * bx.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avg_loss = total_loss / len(x_train_s)
        elapsed_s = time.perf_counter() - t_start
        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f} | {elapsed_s:.1f}s",
            flush=True,
        )
        _emit(
            {
                "stage": "epoch",
                "epoch": epoch,
                "epochs": cfg.epochs,
                "loss": avg_loss,
                "val_macro_f1": val_f1,
                "elapsed_s": elapsed_s,
                "device": device_name,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final evaluation on held-out test set
    with torch.no_grad():
        test_logits = model(torch.from_numpy(x_test_s).to(device))
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)

    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    pr_auc = _compute_macro_pr_auc(y_test, test_probs, num_classes)
    fpr = _compute_binary_fpr(y_test, test_preds)

    print("\n=== Test Evaluation ===", flush=True)
    print(f"Macro-F1:    {macro_f1:.4f}", flush=True)
    print(f"Weighted-F1: {weighted_f1:.4f}", flush=True)
    print(f"Macro PR-AUC:{pr_auc:.4f}", flush=True)
    print(f"Benign FPR:  {fpr:.4f}", flush=True)
    report_str = classification_report(y_test, test_preds, target_names=class_names, digits=4, zero_division=0)
    print(report_str, flush=True)

    # Save artifacts
    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = models_dir / cfg.preprocessor_filename
    onnx_path = models_dir / cfg.onnx_filename
    meta_path = models_dir / cfg.metadata_filename

    preprocessor.save(preproc_path)

    # Remove stale external weights file if present from an older dynamo export
    onnx_data_sidecar = Path(f"{onnx_path}.data")
    if onnx_data_sidecar.exists():
        onnx_data_sidecar.unlink()

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, NUM_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    # Verify ONNX Runtime parity against PyTorch
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = ort_sess.run(["logits"], {"input": x_test_s[:32]})[0]
    with torch.no_grad():
        pt_logits = model_cpu(torch.from_numpy(x_test_s[:32])).numpy()
    max_diff = float(np.max(np.abs(ort_logits - pt_logits)))
    print(f"[ONNX] Exported to {onnx_path} | Max parity diff: {max_diff:.2e}")

    active_count = int(np.sum(preprocessor.active_mask)) if preprocessor.active_mask is not None else NUM_FEATURES
    metadata = {
        "version": "0.1.0",
        "dataset_source": dataset_source,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_features": NUM_FEATURES,
        "active_features": active_count,
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(class_names),
        "metrics": {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_pr_auc": round(pr_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "onnx_max_diff": max_diff,
        },
        "thresholds": {
            "suspicious": cfg.suspicious_threshold,
            "attack": cfg.attack_threshold,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    _emit(
        {
            "stage": "eval",
            "metrics": metadata["metrics"],
            "report": report_str,
            "message": f"Экспорт ONNX завершён (Macro-F1={macro_f1:.4f}, FPR={fpr:.4f})",
        }
    )

    return metadata


if __name__ == "__main__":
    train_and_export()
