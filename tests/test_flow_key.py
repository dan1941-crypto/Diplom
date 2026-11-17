"""Tests for symmetric bidirectional FlowKey and FlowTable."""

from src.capture.flow_key import make_flow_key
from src.capture.flow_table import FlowTable
from src.capture.packet_parser import TH_ACK, TH_FIN, TH_SYN, ParsedPacket


def test_bidirectional_flow_key_is_symmetric() -> None:
    k1 = make_flow_key("192.168.1.10", 51234, "10.0.0.5", 443, 6)
    k2 = make_flow_key("10.0.0.5", 443, "192.168.1.10", 51234, 6)
    assert k1 == k2


def test_flow_table_aggregates_fwd_and_bwd_packets() -> None:
    table = FlowTable(idle_timeout_sec=10.0, active_timeout_sec=60.0)
    p1 = ParsedPacket(
        timestamp=100.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51234,
        dst_port=443,
        proto=6,
        ip_len=60,
        payload_len=0,
        header_len=40,
        tcp_flags=TH_SYN,
        tcp_window=65535,
        ttl=64,
    )
    p2 = ParsedPacket(
        timestamp=100.01,
        src_ip="10.0.0.5",
        dst_ip="192.168.1.10",
        src_port=443,
        dst_port=51234,
        proto=6,
        ip_len=60,
        payload_len=0,
        header_len=40,
        tcp_flags=TH_SYN | TH_ACK,
        tcp_window=64240,
        ttl=56,
    )
    p3 = ParsedPacket(
        timestamp=100.02,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51234,
        dst_port=443,
        proto=6,
        ip_len=200,
        payload_len=160,
        header_len=20,
        tcp_flags=TH_ACK,
        tcp_window=65535,
        ttl=64,
    )

    assert table.process_packet(p1) == []
    assert table.process_packet(p2) == []
    assert table.process_packet(p3) == []
    assert table.active_count == 1

    flows = table.flush_all()
    assert len(flows) == 1
    flow = flows[0]
    assert len(flow.fwd_lengths) == 2
    assert len(flow.bwd_lengths) == 1
    assert flow.init_win_bytes_fwd == 65535
    assert flow.init_win_bytes_bwd == 64240


def test_flow_table_closes_on_two_fins() -> None:
    table = FlowTable()
    p_fin1 = ParsedPacket(1.0, "1.1.1.1", "2.2.2.2", 1234, 80, 6, 52, 0, 32, TH_FIN | TH_ACK, 1024, 64)
    p_fin2 = ParsedPacket(1.01, "2.2.2.2", "1.1.1.1", 80, 1234, 6, 52, 0, 32, TH_FIN | TH_ACK, 1024, 64)

    assert table.process_packet(p_fin1) == []
    completed = table.process_packet(p_fin2)
    assert len(completed) == 1
    assert table.active_count == 0
