"""Stateful bidirectional flow table with timeout and TCP lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.capture.flow_key import FlowKey, make_flow_key
from src.capture.packet_parser import (
    TH_ACK,
    TH_CWR,
    TH_ECE,
    TH_FIN,
    TH_PSH,
    TH_RST,
    TH_SYN,
    TH_URG,
    ParsedPacket,
)


@dataclass(slots=True)
class FlowRecord:
    """Completed or active bidirectional network flow with packet statistics."""

    flow_key: FlowKey
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: int
    start_ts: float
    last_ts: float

    fwd_lengths: list[int] = field(default_factory=list)
    bwd_lengths: list[int] = field(default_factory=list)
    fwd_payload_lengths: list[int] = field(default_factory=list)
    bwd_payload_lengths: list[int] = field(default_factory=list)
    all_lengths: list[int] = field(default_factory=list)

    fwd_timestamps_us: list[float] = field(default_factory=list)
    bwd_timestamps_us: list[float] = field(default_factory=list)
    all_timestamps_us: list[float] = field(default_factory=list)

    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0
    fwd_min_seg_size: int = 0
    fwd_act_data_pkts: int = 0
    init_win_bytes_fwd: int = 0
    init_win_bytes_bwd: int = 0
    has_bwd_init_win: bool = False

    fwd_ttl_sum: int = 0
    bwd_ttl_sum: int = 0

    fin_count: int = 0
    syn_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0
    cwr_count: int = 0
    ece_count: int = 0
    fwd_psh_count: int = 0
    bwd_psh_count: int = 0
    fwd_urg_count: int = 0
    bwd_urg_count: int = 0

    closing: bool = False

    def add_packet(self, pkt: ParsedPacket) -> None:
        """Update flow accumulators with a newly arrived packet."""
        is_fwd = pkt.src_ip == self.src_ip and pkt.src_port == self.src_port
        ts_us = pkt.timestamp * 1_000_000.0
        self.last_ts = max(self.last_ts, pkt.timestamp)

        self.all_lengths.append(pkt.ip_len)
        self.all_timestamps_us.append(ts_us)

        if is_fwd:
            self.fwd_lengths.append(pkt.ip_len)
            self.fwd_payload_lengths.append(pkt.payload_len)
            self.fwd_timestamps_us.append(ts_us)
            self.fwd_header_bytes += pkt.header_len
            self.fwd_ttl_sum += pkt.ttl
            if pkt.header_len > 0:
                if self.fwd_min_seg_size == 0 or pkt.header_len < self.fwd_min_seg_size:
                    self.fwd_min_seg_size = pkt.header_len
            if pkt.payload_len > 0:
                self.fwd_act_data_pkts += 1
            if len(self.fwd_lengths) == 1:
                self.init_win_bytes_fwd = pkt.tcp_window
        else:
            self.bwd_lengths.append(pkt.ip_len)
            self.bwd_payload_lengths.append(pkt.payload_len)
            self.bwd_timestamps_us.append(ts_us)
            self.bwd_header_bytes += pkt.header_len
            self.bwd_ttl_sum += pkt.ttl
            if not self.has_bwd_init_win:
                self.init_win_bytes_bwd = pkt.tcp_window
                self.has_bwd_init_win = True

        flags = pkt.tcp_flags
        if flags:
            if flags & TH_FIN:
                self.fin_count += 1
                self.closing = True
            if flags & TH_SYN:
                self.syn_count += 1
            if flags & TH_RST:
                self.rst_count += 1
                self.closing = True
            if flags & TH_PSH:
                self.psh_count += 1
                if is_fwd:
                    self.fwd_psh_count += 1
                else:
                    self.bwd_psh_count += 1
            if flags & TH_ACK:
                self.ack_count += 1
            if flags & TH_URG:
                self.urg_count += 1
                if is_fwd:
                    self.fwd_urg_count += 1
                else:
                    self.bwd_urg_count += 1
            if flags & TH_CWR:
                self.cwr_count += 1
            if flags & TH_ECE:
                self.ece_count += 1


class FlowTable:
    """Maintains active bidirectional flows and emits completed FlowRecords."""

    def __init__(
        self,
        idle_timeout_sec: float = 15.0,
        active_timeout_sec: float = 120.0,
        closing_grace_sec: float = 2.0,
        max_flows: int = 50_000,
    ) -> None:
        self.idle_timeout_sec = idle_timeout_sec
        self.active_timeout_sec = active_timeout_sec
        self.closing_grace_sec = closing_grace_sec
        self.max_flows = max_flows
        self._active: dict[FlowKey, FlowRecord] = {}

    @property
    def active_count(self) -> int:
        """Number of currently active flows."""
        return len(self._active)

    def process_packet(self, pkt: ParsedPacket) -> list[FlowRecord]:
        """Insert a packet into the flow table and return any newly completed flows."""
        completed: list[FlowRecord] = []
        key = make_flow_key(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.proto)

        flow = self._active.get(key)
        if flow is not None:
            is_new_syn = bool((pkt.tcp_flags & TH_SYN) and not (pkt.tcp_flags & TH_ACK))
            if (
                pkt.timestamp - flow.last_ts >= self.idle_timeout_sec
                or pkt.timestamp - flow.start_ts >= self.active_timeout_sec
                or (flow.closing and (is_new_syn or pkt.timestamp - flow.last_ts >= self.closing_grace_sec))
            ):
                completed.append(self._active.pop(key))
                flow = None

        if flow is None:
            if len(self._active) >= self.max_flows:
                completed.extend(self.flush_expired(pkt.timestamp))
                if len(self._active) >= self.max_flows:
                    oldest_key = next(iter(self._active))
                    completed.append(self._active.pop(oldest_key))

            flow = FlowRecord(
                flow_key=key,
                src_ip=pkt.src_ip,
                src_port=pkt.src_port,
                dst_ip=pkt.dst_ip,
                dst_port=pkt.dst_port,
                proto=pkt.proto,
                start_ts=pkt.timestamp,
                last_ts=pkt.timestamp,
            )
            self._active[key] = flow

        flow.add_packet(pkt)

        # Enter closing grace window on RST or bidirectional FIN+ACK so trailing teardown ACKs
        # are absorbed into the same flow instead of spawning 1-packet zombie flows.
        if (pkt.tcp_flags & TH_RST) or (flow.fin_count >= 2 and (pkt.tcp_flags & TH_ACK)):
            flow.closing = True

        return completed

    def flush_expired(self, now_ts: float) -> list[FlowRecord]:
        """Remove and return all flows that have exceeded idle, active, or closing timeouts."""
        expired_keys = [
            key
            for key, flow in self._active.items()
            if (
                now_ts - flow.last_ts >= self.idle_timeout_sec
                or now_ts - flow.start_ts >= self.active_timeout_sec
                or (flow.closing and now_ts - flow.last_ts >= self.closing_grace_sec)
            )
        ]
        return [self._active.pop(k) for k in expired_keys]

    def flush_all(self) -> list[FlowRecord]:
        """Drain and return all remaining active flows."""
        flows = list(self._active.values())
        self._active.clear()
        return flows
