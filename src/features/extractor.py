"""Single shared bidirectional flow feature extractor (Train = Inference)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.capture.flow_table import FlowRecord

FEATURE_NAMES: tuple[str, ...] = (
    # Protocol & Destination Port buckets/flags (14) - NO raw IPs or src_port
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
    "dst_port_well_known",
    "dst_port_registered",
    "dst_port_dynamic",
    "dst_port_http",
    "dst_port_https",
    "dst_port_dns",
    "dst_port_ssh",
    "dst_port_ftp",
    "dst_port_rdp",
    "dst_port_smb",
    # Duration & Rates (5) - microseconds
    "flow_duration_us",
    "flow_bytes_per_s",
    "flow_pkts_per_s",
    "fwd_pkts_per_s",
    "bwd_pkts_per_s",
    # Counts & Ratios (8)
    "total_fwd_pkts",
    "total_bwd_pkts",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_fwd_payload_bytes",
    "total_bwd_payload_bytes",
    "down_up_ratio",
    "bwd_fwd_bytes_ratio",
    # Packet Length Stats (16)
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "avg_pkt_size",
    "avg_fwd_seg_size",
    "avg_bwd_seg_size",
    # IAT Stats in microseconds (14)
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    # Headers, Windows, TTL, Flags (20)
    "fwd_header_len",
    "bwd_header_len",
    "fwd_min_seg_size",
    "fwd_act_data_pkts",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "fwd_ttl_mean",
    "bwd_ttl_mean",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "urg_flag_cnt",
    "cwr_flag_cnt",
    "ece_flag_cnt",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    # Active / Idle burst stats in microseconds (8)
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
)

NUM_FEATURES = len(FEATURE_NAMES)
ACTIVE_IDLE_THRESHOLD_US = 1_000_000.0  # 1 second gap separates active bursts


def _stats_4(values: Sequence[float] | Sequence[int]) -> tuple[float, float, float, float]:
    """Return (min, max, mean, std) for a sequence, or zeros if empty."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    if n == 1:
        v = float(arr[0])
        return v, v, v, 0.0
    return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std(ddof=1))


def _iat_stats(timestamps_us: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Return (total, mean, std, max, min) of inter-arrival times in microseconds."""
    if len(timestamps_us) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ts = np.asarray(timestamps_us, dtype=np.float64)
    diffs = np.maximum(0.0, np.diff(ts))
    if len(diffs) == 1:
        d = float(diffs[0])
        return d, d, 0.0, d, d
    return (
        float(diffs.sum()),
        float(diffs.mean()),
        float(diffs.std(ddof=1)),
        float(diffs.max()),
        float(diffs.min()),
    )


def _active_idle_stats(
    timestamps_us: Sequence[float],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute Active and Idle burst (mean, std, max, min) in microseconds."""
    if len(timestamps_us) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    active_durations: list[float] = []
    idle_durations: list[float] = []

    burst_start = timestamps_us[0]
    prev_ts = timestamps_us[0]

    for ts in timestamps_us[1:]:
        gap = max(0.0, ts - prev_ts)
        if gap > ACTIVE_IDLE_THRESHOLD_US:
            active_durations.append(max(0.0, prev_ts - burst_start))
            idle_durations.append(gap)
            burst_start = ts
        prev_ts = ts

    if idle_durations:
        active_durations.append(max(0.0, prev_ts - burst_start))

    a_min, a_max, a_mean, a_std = _stats_4(active_durations)
    i_min, i_max, i_mean, i_std = _stats_4(idle_durations)
    return a_mean, a_std, a_max, a_min, i_mean, i_std, i_max, i_min


def extract_flow_features(flow: FlowRecord) -> np.ndarray:
    """Extract a fixed-length float32 feature vector from a FlowRecord."""
    proto = flow.proto
    proto_tcp = 1.0 if proto == 6 else 0.0
    proto_udp = 1.0 if proto == 17 else 0.0
    proto_icmp = 1.0 if proto in (1, 58) else 0.0
    proto_other = 1.0 if (proto not in (1, 6, 17, 58)) else 0.0

    dport = flow.dst_port
    dst_well_known = 1.0 if 0 <= dport <= 1023 else 0.0
    dst_registered = 1.0 if 1024 <= dport <= 49151 else 0.0
    dst_dynamic = 1.0 if dport >= 49152 else 0.0
    dst_http = 1.0 if dport in (80, 8080, 8000) else 0.0
    dst_https = 1.0 if dport in (443, 8443) else 0.0
    dst_dns = 1.0 if dport == 53 else 0.0
    dst_ssh = 1.0 if dport == 22 else 0.0
    dst_ftp = 1.0 if dport in (20, 21) else 0.0
    dst_rdp = 1.0 if dport == 3389 else 0.0
    dst_smb = 1.0 if dport == 445 else 0.0

    duration_us = max(0.0, (flow.last_ts - flow.start_ts) * 1_000_000.0)

    n_fwd = len(flow.fwd_lengths)
    n_bwd = len(flow.bwd_lengths)
    n_all = len(flow.all_lengths)

    total_fwd_bytes = float(sum(flow.fwd_lengths))
    total_bwd_bytes = float(sum(flow.bwd_lengths))
    total_bytes = total_fwd_bytes + total_bwd_bytes

    total_fwd_payload = float(sum(flow.fwd_payload_lengths))
    total_bwd_payload = float(sum(flow.bwd_payload_lengths))

    if n_all > 1 and duration_us >= 100.0:
        duration_sec = duration_us / 1_000_000.0
        flow_bytes_s = total_bytes / duration_sec
        flow_pkts_s = n_all / duration_sec
        fwd_pkts_s = n_fwd / duration_sec
        bwd_pkts_s = n_bwd / duration_sec
    else:
        flow_bytes_s = 0.0
        flow_pkts_s = 0.0
        fwd_pkts_s = 0.0
        bwd_pkts_s = 0.0

    down_up_ratio = float(n_bwd) / float(n_fwd) if n_fwd > 0 else 0.0
    bwd_fwd_bytes_ratio = total_bwd_bytes / total_fwd_bytes if total_fwd_bytes > 0.0 else 0.0

    fwd_min, fwd_max, fwd_mean, fwd_std = _stats_4(flow.fwd_lengths)
    bwd_min, bwd_max, bwd_mean, bwd_std = _stats_4(flow.bwd_lengths)
    all_min, all_max, all_mean, all_std = _stats_4(flow.all_lengths)
    all_var = all_std * all_std

    avg_pkt_size = total_bytes / n_all if n_all > 0 else 0.0
    avg_fwd_seg = total_fwd_bytes / n_fwd if n_fwd > 0 else 0.0
    avg_bwd_seg = total_bwd_bytes / n_bwd if n_bwd > 0 else 0.0

    _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _iat_stats(flow.all_timestamps_us)
    fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _iat_stats(flow.fwd_timestamps_us)
    bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _iat_stats(flow.bwd_timestamps_us)

    fwd_ttl_mean = float(flow.fwd_ttl_sum) / n_fwd if n_fwd > 0 else 0.0
    bwd_ttl_mean = float(flow.bwd_ttl_sum) / n_bwd if n_bwd > 0 else 0.0

    act_mean, act_std, act_max, act_min, idl_mean, idl_std, idl_max, idl_min = _active_idle_stats(
        flow.all_timestamps_us
    )

    vec = np.array(
        [
            proto_tcp,
            proto_udp,
            proto_icmp,
            proto_other,
            dst_well_known,
            dst_registered,
            dst_dynamic,
            dst_http,
            dst_https,
            dst_dns,
            dst_ssh,
            dst_ftp,
            dst_rdp,
            dst_smb,
            duration_us,
            flow_bytes_s,
            flow_pkts_s,
            fwd_pkts_s,
            bwd_pkts_s,
            float(n_fwd),
            float(n_bwd),
            total_fwd_bytes,
            total_bwd_bytes,
            total_fwd_payload,
            total_bwd_payload,
            down_up_ratio,
            bwd_fwd_bytes_ratio,
            fwd_max,
            fwd_min,
            fwd_mean,
            fwd_std,
            bwd_max,
            bwd_min,
            bwd_mean,
            bwd_std,
            all_min,
            all_max,
            all_mean,
            all_std,
            all_var,
            avg_pkt_size,
            avg_fwd_seg,
            avg_bwd_seg,
            flow_iat_mean,
            flow_iat_std,
            flow_iat_max,
            flow_iat_min,
            fwd_iat_tot,
            fwd_iat_mean,
            fwd_iat_std,
            fwd_iat_max,
            fwd_iat_min,
            bwd_iat_tot,
            bwd_iat_mean,
            bwd_iat_std,
            bwd_iat_max,
            bwd_iat_min,
            float(flow.fwd_header_bytes),
            float(flow.bwd_header_bytes),
            float(flow.fwd_min_seg_size),
            float(flow.fwd_act_data_pkts),
            float(flow.init_win_bytes_fwd),
            float(flow.init_win_bytes_bwd),
            fwd_ttl_mean,
            bwd_ttl_mean,
            1.0 if flow.fin_count > 0 else 0.0,
            1.0 if flow.syn_count > 0 else 0.0,
            1.0 if flow.rst_count > 0 else 0.0,
            1.0 if flow.psh_count > 0 else 0.0,
            1.0 if flow.ack_count > 0 else 0.0,
            1.0 if flow.urg_count > 0 else 0.0,
            1.0 if flow.cwr_count > 0 else 0.0,
            1.0 if flow.ece_count > 0 else 0.0,
            float(flow.fwd_psh_count),
            float(flow.bwd_psh_count),
            float(flow.fwd_urg_count),
            float(flow.bwd_urg_count),
            act_mean,
            act_std,
            act_max,
            act_min,
            idl_mean,
            idl_std,
            idl_max,
            idl_min,
        ],
        dtype=np.float32,
    )
    # Replace any non-finite values defensively
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e9, neginf=0.0)
    return vec


def extract_batch_features(flows: Sequence[FlowRecord]) -> np.ndarray:
    """Extract a 2D float32 array of shape (N, NUM_FEATURES) from flows."""
    if not flows:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)
    return np.vstack([extract_flow_features(f) for f in flows]).astype(np.float32, copy=False)


def format_feature_value(name: str, value: float) -> str:
    """Human-readable formatting for UI flow inspection."""
    if math.isnan(value):
        return "0"
    if name.endswith("_us") or "iat" in name or name.startswith(("active_", "idle_")):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} s"
        if value >= 1_000:
            return f"{value / 1_000:.2f} ms"
        return f"{value:.1f} us"
    if "per_s" in name:
        return f"{value:,.1f}/s"
    if value == int(value) and abs(value) < 1e9:
        return str(int(value))
    return f"{value:,.2f}"
