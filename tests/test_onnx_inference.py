"""End-to-end tests for training, ONNX export, inference, and real PCAPNG decoding."""

from dataclasses import replace
from pathlib import Path

from src.capture.flow_table import FlowTable
from src.capture.packet_parser import iter_pcap_packets
from src.config import load_config
from src.ml.inference import FlowInferenceEngine
from src.ml.train import train_and_export


def test_train_export_and_onnx_inference(tmp_path: Path) -> None:
    base_cfg = load_config()
    test_cfg = replace(
        base_cfg,
        models_dir=tmp_path / "models",
        epochs=4,
    )
    meta = train_and_export(test_cfg, total_samples=600)
    assert meta["metrics"]["macro_f1"] > 0.5
    assert meta["metrics"]["onnx_max_diff"] < 1e-4

    engine = FlowInferenceEngine(test_cfg)

    # Run inference on real packets from legacy_reference/test.pcapng (first 500 packets)
    table = FlowTable()
    for idx, pkt in enumerate(iter_pcap_packets(base_cfg.default_pcap)):
        table.process_packet(pkt)
        if idx >= 500:
            break
    flows = table.flush_all()
    assert len(flows) > 0

    preds = engine.predict_batch(flows[:20])
    assert len(preds) == min(20, len(flows))
    for p in preds:
        assert p.severity in ("Normal", "Suspicious", "Attack")
        assert 0.0 <= p.confidence <= 1.0
        assert len(p.top_features) > 0
