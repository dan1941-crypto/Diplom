"""ONNX Runtime inference engine with severity calibration and feature attribution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from src.capture.flow_table import FlowRecord
from src.config import AppConfig, load_config
from src.features.extractor import FEATURE_NAMES, extract_batch_features, format_feature_value
from src.features.preprocessor import FlowPreprocessor


@dataclass(slots=True)
class FlowPrediction:
    """Inference result for a single network flow."""

    flow: FlowRecord
    predicted_class: str
    class_index: int
    confidence: float
    attack_score: float
    severity: str  # "Normal" | "Suspicious" | "Attack"
    probabilities: dict[str, float]
    top_features: list[tuple[str, str, float]]  # (feature_name, formatted_val, z_magnitude)
    latency_ms: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class FlowInferenceEngine:
    """Runs batch inference on FlowRecord objects using FlowPreprocessor + ONNX Runtime."""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or load_config()
        models_dir = Path(self.cfg.models_dir)
        self.onnx_path = models_dir / self.cfg.onnx_filename
        self.preproc_path = models_dir / self.cfg.preprocessor_filename
        self.meta_path = models_dir / self.cfg.metadata_filename

        if not self.onnx_path.exists() or not self.preproc_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Run `python -m src.ml.train` first."
            )

        self.preprocessor = FlowPreprocessor.load(self.preproc_path)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.class_names: list[str] = list(self.metadata.get("class_names", []))
        else:
            self.metadata = {}
            self.class_names = ["BENIGN", "DDoS/DoS", "PortScan", "BruteForce", "Exfiltration"]

        self.suspicious_threshold = self.cfg.suspicious_threshold
        self.attack_threshold = self.cfg.attack_threshold

    def predict_batch(self, flows: Sequence[FlowRecord], top_k: int = 8) -> list[FlowPrediction]:
        """Classify a batch of FlowRecord objects and return structured predictions."""
        if not flows:
            return []

        t0 = time.perf_counter()
        raw_x = extract_batch_features(flows)
        scaled_x = self.preprocessor.transform(raw_x)
        logits = self.session.run(["logits"], {"input": scaled_x})[0]
        probs = _softmax(logits)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_flow_ms = elapsed_ms / len(flows)

        results: list[FlowPrediction] = []
        for i, flow in enumerate(flows):
            p = probs[i]
            cls_idx = int(np.argmax(p))
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class_{cls_idx}"
            attack_score = float(1.0 - p[0])

            if attack_score >= self.attack_threshold and cls_idx != 0:
                severity = "Attack"
            elif attack_score >= self.suspicious_threshold:
                severity = "Suspicious"
            else:
                severity = "Normal"

            confidence = float(p[cls_idx])

            # Top-K deviating features by absolute RobustScaler magnitude
            abs_scaled = np.abs(scaled_x[i])
            top_indices = np.argsort(abs_scaled)[::-1][:top_k]
            top_feats = [
                (
                    FEATURE_NAMES[idx],
                    format_feature_value(FEATURE_NAMES[idx], float(raw_x[i, idx])),
                    float(abs_scaled[idx]),
                )
                for idx in top_indices
            ]

            prob_map = {
                (self.class_names[j] if j < len(self.class_names) else f"Class_{j}"): float(p[j])
                for j in range(len(p))
            }

            results.append(
                FlowPrediction(
                    flow=flow,
                    predicted_class=cls_name,
                    class_index=cls_idx,
                    confidence=confidence,
                    attack_score=attack_score,
                    severity=severity,
                    probabilities=prob_map,
                    top_features=top_feats,
                    latency_ms=per_flow_ms,
                )
            )

        return results
