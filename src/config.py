"""Configuration loader and relative path utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def resolve_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class AppConfig:
    """Typed application configuration."""

    raw: dict[str, Any]
    models_dir: Path
    data_dir: Path
    default_pcap: Path
    idle_timeout_sec: float
    active_timeout_sec: float
    max_flows: int
    capture_batch_size: int
    capture_batch_interval_ms: int
    clip_max: float
    service_ports: tuple[int, ...]
    hidden_dim: int
    num_blocks: int
    dropout: float
    epochs: int
    train_batch_size: int
    learning_rate: float
    weight_decay: float
    focal_gamma: float
    val_split: float
    test_split: float
    random_seed: int
    suspicious_threshold: float
    attack_threshold: float
    onnx_filename: str
    preprocessor_filename: str
    metadata_filename: str


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration from disk."""
    path = resolve_path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_cfg = raw.get("paths", {})
    cap_cfg = raw.get("capture", {})
    feat_cfg = raw.get("features", {})
    model_cfg = raw.get("model", {})
    inf_cfg = raw.get("inference", {})

    return AppConfig(
        raw=raw,
        models_dir=resolve_path(paths_cfg.get("models_dir", "models")),
        data_dir=resolve_path(paths_cfg.get("data_dir", "data")),
        default_pcap=resolve_path(paths_cfg.get("default_pcap", "legacy_reference/test.pcapng")),
        idle_timeout_sec=float(cap_cfg.get("idle_timeout_sec", 15.0)),
        active_timeout_sec=float(cap_cfg.get("active_timeout_sec", 120.0)),
        max_flows=int(cap_cfg.get("max_flows", 50000)),
        capture_batch_size=int(cap_cfg.get("batch_size", 64)),
        capture_batch_interval_ms=int(cap_cfg.get("batch_interval_ms", 250)),
        clip_max=float(feat_cfg.get("clip_max", 1_000_000_000.0)),
        service_ports=tuple(int(p) for p in feat_cfg.get("service_ports", [])),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_blocks=int(model_cfg.get("num_blocks", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        epochs=int(model_cfg.get("epochs", 25)),
        train_batch_size=int(model_cfg.get("batch_size", 256)),
        learning_rate=float(model_cfg.get("learning_rate", 0.001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0001)),
        focal_gamma=float(model_cfg.get("focal_gamma", 2.0)),
        val_split=float(model_cfg.get("val_split", 0.15)),
        test_split=float(model_cfg.get("test_split", 0.15)),
        random_seed=int(model_cfg.get("random_seed", 42)),
        suspicious_threshold=float(inf_cfg.get("suspicious_threshold", 0.45)),
        attack_threshold=float(inf_cfg.get("attack_threshold", 0.70)),
        onnx_filename=str(inf_cfg.get("onnx_filename", "nids_resnet.onnx")),
        preprocessor_filename=str(inf_cfg.get("preprocessor_filename", "preprocessor.joblib")),
        metadata_filename=str(inf_cfg.get("metadata_filename", "metadata.json")),
    )
