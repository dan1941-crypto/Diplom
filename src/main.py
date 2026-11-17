"""Autonomous entry point for NIDS Traffic Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import load_config
from src.ml.inference import FlowInferenceEngine
from src.ui.main_window import run_gui


def main() -> int:
    """Ensure ONNX model artifacts exist (auto-training if needed) and launch the GUI."""
    cfg = load_config()
    onnx_path = Path(cfg.models_dir) / cfg.onnx_filename
    preproc_path = Path(cfg.models_dir) / cfg.preprocessor_filename

    if not onnx_path.exists() or not preproc_path.exists():
        print("[Bootstrap] Model artifacts not found. Running autonomous training & ONNX export...")
        from src.ml.train import train_and_export

        train_and_export(cfg)

    engine = FlowInferenceEngine(cfg)
    return run_gui(cfg, engine)


if __name__ == "__main__":
    sys.exit(main())
