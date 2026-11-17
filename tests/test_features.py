"""Tests for unified bidirectional feature extraction."""

import numpy as np

from src.capture.flow_key import make_flow_key
from src.capture.flow_table import FlowRecord
from src.capture.packet_parser import TH_ACK, TH_PSH, TH_SYN, ParsedPacket
from src.features.extractor import FEATURE_NAMES, NUM_FEATURES, extract_flow_features


def test_feature_vector_shape_and_units() -> None:
    key = make_flow_key("192.168.1.2", 50000, "10.0.0.1", 443, 6)
    flow = FlowRecord(
        flow_key=key,
        src_ip="192.168.1.2",
        src_port=50000,
        dst_ip="10.0.0.1",
        dst_port=443,
        proto=6,
        start_ts=10.0,
        last_ts=10.0,
    )
    # 0.0s, 0.05s, and 1.5s (> 1.0s gap to trigger Active/Idle stats)
    flow.add_packet(ParsedPacket(10.0, "192.168.1.2", "10.0.0.1", 50000, 443, 6, 60, 0, 40, TH_SYN, 65535, 64))
    flow.add_packet(ParsedPacket(10.05, "10.0.0.1", "192.168.1.2", 443, 50000, 6, 1200, 1160, 20, TH_ACK | TH_PSH, 64240, 56))
    flow.add_packet(ParsedPacket(11.55, "192.168.1.2", "10.0.0.1", 50000, 443, 6, 120, 80, 20, TH_ACK | TH_PSH, 65535, 64))

    vec = extract_flow_features(flow)
    assert vec.shape == (NUM_FEATURES,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))

    feat_map = dict(zip(FEATURE_NAMES, vec.tolist()))
    assert feat_map["proto_tcp"] == 1.0
    assert feat_map["dst_port_https"] == 1.0
    assert abs(feat_map["flow_duration_us"] - 1_550_000.0) < 1.0
    assert feat_map["down_up_ratio"] == 0.5
    assert feat_map["fwd_header_len"] == 60.0
    assert feat_map["bwd_header_len"] == 20.0
    assert feat_map["init_win_bytes_fwd"] == 65535.0
    assert feat_map["init_win_bytes_bwd"] == 64240.0
    assert feat_map["idle_mean"] > 1_000_000.0
