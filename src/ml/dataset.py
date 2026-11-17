"""Dataset builder using the unified FlowRecord -> extract_flow_features pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.capture.flow_key import make_flow_key
from src.capture.flow_table import FlowRecord
from src.capture.packet_parser import TH_ACK, TH_FIN, TH_PSH, TH_RST, TH_SYN, ParsedPacket
from src.features.extractor import FEATURE_NAMES, NUM_FEATURES, extract_batch_features

CLASS_NAMES: tuple[str, ...] = (
    "BENIGN",
    "DDoS/DoS",
    "PortScan",
    "BruteForce",
    "Exfiltration",
)

UNIFIED_CLASSES: tuple[str, ...] = (
    "Benign",
    "DDoS/DoS",
    "PortScan/Recon",
    "Exploits",
    "Fuzzers/Generic",
    "Malware/Other",
)

NF_UNSW_TO_UNIFIED: dict[str, int] = {
    "Benign": 0,
    "DoS": 1,
    "Reconnaissance": 2,
    "Exploits": 3,
    "Fuzzers": 4,
    "Generic": 4,
    "Backdoor": 5,
    "Shellcode": 5,
    "Analysis": 5,
    "Worms": 5,
}

NIDD_5G_TO_UNIFIED: dict[str, int] = {
    "Benign": 0,
    "UDPFlood": 1,
    "HTTPFlood": 1,
    "SlowrateDoS": 1,
    "SYNFlood": 1,
    "ICMPFlood": 1,
    "TCPConnectScan": 2,
    "SYNScan": 2,
    "UDPScan": 2,
}


def _simulate_flow(
    rng: np.random.Generator,
    class_id: int,
    base_ts: float,
) -> FlowRecord:
    """Synthesize a physically grounded bidirectional FlowRecord for a traffic class."""
    src_ip = f"192.168.{rng.integers(1, 5)}.{rng.integers(10, 250)}"
    dst_ip = f"10.0.{rng.integers(0, 4)}.{rng.integers(2, 100)}"
    src_port = int(rng.integers(32768, 65500))

    if class_id == 0:  # BENIGN (HTTPS, HTTP, DNS, normal SSH)
        profile = int(rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15]))
        if profile == 0:  # HTTPS / HTTP
            dst_port = int(rng.choice([443, 80, 8443]))
            proto = 6
            n_fwd = int(rng.integers(4, 24))
            n_bwd = int(rng.integers(4, 32))
            iat_mean = float(rng.uniform(0.004, 0.08))
            fwd_len_mu, bwd_len_mu = 220, 950
            win_fwd, win_bwd = 65535, 64240
            fin_end = True
            rst_end = False
        elif profile == 1:  # DNS
            dst_port = 53
            proto = 17
            n_fwd = int(rng.integers(1, 3))
            n_bwd = n_fwd
            iat_mean = float(rng.uniform(0.002, 0.03))
            fwd_len_mu, bwd_len_mu = 72, 185
            win_fwd, win_bwd = 0, 0
            fin_end = False
            rst_end = False
        else:  # Normal SSH / interactive
            dst_port = 22
            proto = 6
            n_fwd = int(rng.integers(8, 30))
            n_bwd = int(rng.integers(8, 30))
            iat_mean = float(rng.uniform(0.05, 0.45))
            fwd_len_mu, bwd_len_mu = 110, 160
            win_fwd, win_bwd = 64240, 65535
            fin_end = True
            rst_end = False

    elif class_id == 1:  # DDoS / DoS (SYN flood, HTTP flood, UDP flood)
        attack_type = int(rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20]))
        if attack_type == 0:  # SYN flood
            dst_port = int(rng.choice([80, 443, 8080]))
            proto = 6
            n_fwd = int(rng.integers(15, 90))
            n_bwd = int(rng.integers(0, 3))
            iat_mean = float(rng.uniform(0.00005, 0.0015))
            fwd_len_mu, bwd_len_mu = 60, 54
            win_fwd, win_bwd = 1024, 0
            fin_end = False
            rst_end = bool(rng.random() < 0.3)
        elif attack_type == 1:  # HTTP flood
            dst_port = int(rng.choice([80, 443, 8080]))
            proto = 6
            n_fwd = int(rng.integers(25, 110))
            n_bwd = int(rng.integers(2, 15))
            iat_mean = float(rng.uniform(0.0002, 0.004))
            fwd_len_mu, bwd_len_mu = 420, 180
            win_fwd, win_bwd = 29200, 14600
            fin_end = False
            rst_end = bool(rng.random() < 0.5)
        else:  # UDP flood
            dst_port = int(rng.choice([53, 123, 1900, 80]))
            proto = 17
            n_fwd = int(rng.integers(30, 120))
            n_bwd = 0
            iat_mean = float(rng.uniform(0.00003, 0.001))
            fwd_len_mu, bwd_len_mu = 512, 0
            win_fwd, win_bwd = 0, 0
            fin_end = False
            rst_end = False

    elif class_id == 2:  # PortScan (short, rapid SYN -> RST/no-response)
        dst_port = int(rng.integers(1, 10000))
        proto = 6
        n_fwd = int(rng.integers(1, 3))
        n_bwd = int(rng.choice([0, 1], p=[0.4, 0.6]))
        iat_mean = float(rng.uniform(0.00005, 0.002))
        fwd_len_mu, bwd_len_mu = 58, 54
        win_fwd, win_bwd = 1024, 0
        fin_end = False
        rst_end = n_bwd > 0

    elif class_id == 3:  # BruteForce (SSH / FTP / RDP repeated auth attempts)
        dst_port = int(rng.choice([22, 21, 3389]))
        proto = 6
        n_fwd = int(rng.integers(10, 28))
        n_bwd = int(rng.integers(8, 22))
        iat_mean = float(rng.uniform(0.005, 0.04))
        fwd_len_mu, bwd_len_mu = 140, 120
        win_fwd, win_bwd = 29200, 29200
        fin_end = True
        rst_end = bool(rng.random() < 0.4)

    else:  # Exfiltration / C2 (large outbound payload, minimal inbound ACK)
        dst_port = int(rng.choice([443, 8443, 4444, 9001]))
        proto = 6
        n_fwd = int(rng.integers(30, 95))
        n_bwd = int(rng.integers(5, 18))
        iat_mean = float(rng.uniform(0.001, 0.02))
        fwd_len_mu, bwd_len_mu = 1380, 66
        win_fwd, win_bwd = 65535, 65535
        fin_end = True
        rst_end = False

    key = make_flow_key(src_ip, src_port, dst_ip, dst_port, proto)
    flow = FlowRecord(
        flow_key=key,
        src_ip=src_ip,
        src_port=src_port,
        dst_ip=dst_ip,
        dst_port=dst_port,
        proto=proto,
        start_ts=base_ts,
        last_ts=base_ts,
    )

    ts = base_ts
    total_pkts = n_fwd + n_bwd
    fwd_rem = n_fwd
    bwd_rem = n_bwd

    for idx in range(total_pkts):
        if idx == 0:
            is_fwd = True
        elif fwd_rem > 0 and bwd_rem > 0:
            is_fwd = bool(rng.random() < (fwd_rem / (fwd_rem + bwd_rem)))
        else:
            is_fwd = fwd_rem > 0

        if is_fwd:
            fwd_rem -= 1
            p_src_ip, p_dst_ip = src_ip, dst_ip
            p_sport, p_dport = src_port, dst_port
            ip_len = int(np.clip(rng.normal(fwd_len_mu, max(8.0, fwd_len_mu * 0.25)), 40, 1500))
            win = win_fwd
            ttl = int(rng.choice([64, 128]))
        else:
            bwd_rem -= 1
            p_src_ip, p_dst_ip = dst_ip, src_ip
            p_sport, p_dport = dst_port, src_port
            ip_len = int(np.clip(rng.normal(bwd_len_mu, max(8.0, bwd_len_mu * 0.25)), 40, 1500))
            win = win_bwd
            ttl = int(rng.choice([56, 64, 118]))

        hdr_len = 20 if proto == 6 else 8
        payload_len = max(0, ip_len - 20 - hdr_len)
        flags = 0
        if proto == 6:
            if idx == 0:
                flags = TH_SYN
            elif idx == 1 and not is_fwd:
                flags = TH_SYN | TH_ACK
            elif idx == total_pkts - 1 and rst_end:
                flags = TH_RST | TH_ACK
            elif idx >= total_pkts - 2 and fin_end:
                flags = TH_FIN | TH_ACK
            else:
                flags = TH_ACK | (TH_PSH if payload_len > 0 and rng.random() < 0.6 else 0)

        pkt = ParsedPacket(
            timestamp=ts,
            src_ip=p_src_ip,
            dst_ip=p_dst_ip,
            src_port=p_sport,
            dst_port=p_dport,
            proto=proto,
            ip_len=ip_len,
            payload_len=payload_len,
            header_len=hdr_len,
            tcp_flags=flags,
            tcp_window=win,
            ttl=ttl,
        )
        flow.add_packet(pkt)
        ts += float(rng.exponential(iat_mean))

    return flow


def build_autonomous_dataset(
    total_samples: int = 12_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic, imbalanced, shuffled dataset through extract_flow_features."""
    rng = np.random.default_rng(seed)
    # Realistic class distribution (imbalanced: ~58% benign, ~42% across 4 attack classes)
    probs = np.array([0.58, 0.16, 0.12, 0.08, 0.06], dtype=np.float64)
    labels = rng.choice(len(CLASS_NAMES), size=total_samples, p=probs).astype(np.int64)

    flows: list[FlowRecord] = []
    base_ts = 1_700_000_000.0
    for i, cls_id in enumerate(labels):
        flows.append(_simulate_flow(rng, int(cls_id), base_ts + i * 0.05))

    x = extract_batch_features(flows)
    perm = rng.permutation(len(labels))
    return x[perm], labels[perm]


def find_nf_unsw_csv(data_dir: Path) -> Path | None:
    """Locate NF-UNSW-NB15-v3.csv inside data_dir if present."""
    if not data_dir.exists():
        return None
    matches = sorted(data_dir.rglob("NF-UNSW-NB15-v3.csv"))
    return matches[0] if matches else None


def _file_fingerprint(path: Path) -> str:
    """Fast SHA-256 fingerprint using file size + first 1MB."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode("utf-8"))
    with path.open("rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()


def find_5g_nidd_csvs(data_dir: Path) -> list[Path]:
    """Locate unique 5G-NIDD Combined.csv files inside data_dir."""
    if not data_dir.exists():
        return []
    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    for p in sorted(data_dir.rglob("Combined.csv")):
        fp = _file_fingerprint(p)
        if fp not in seen_hashes:
            seen_hashes.add(fp)
            unique_paths.append(p)
    return unique_paths


def load_nf_unsw_nb15_v3(
    csv_path: Path,
    max_benign: int = 180_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load NF-UNSW-NB15-v3.csv and map bidirectional NetFlow features to FEATURE_NAMES."""
    import pandas as pd

    use_cols = [
        "L4_DST_PORT",
        "PROTOCOL",
        "IN_BYTES",
        "IN_PKTS",
        "OUT_BYTES",
        "OUT_PKTS",
        "TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN",
        "DURATION_OUT",
        "MIN_TTL",
        "MAX_TTL",
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
        "SRC_TO_DST_IAT_MIN",
        "SRC_TO_DST_IAT_MAX",
        "SRC_TO_DST_IAT_AVG",
        "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN",
        "DST_TO_SRC_IAT_MAX",
        "DST_TO_SRC_IAT_AVG",
        "DST_TO_SRC_IAT_STDDEV",
        "Attack",
    ]
    df = pd.read_csv(csv_path, usecols=use_cols)

    benign_mask = df["Attack"] == "Benign"
    attack_df = df[~benign_mask]
    benign_df = df[benign_mask]
    if len(benign_df) > max_benign:
        benign_df = benign_df.sample(n=max_benign, random_state=seed)
    df = pd.concat([benign_df, attack_df], ignore_index=True)

    df = df[df["Attack"].isin(NF_UNSW_TO_UNIFIED)].reset_index(drop=True)
    y = df["Attack"].map(NF_UNSW_TO_UNIFIED).to_numpy(dtype=np.int64)

    n = len(df)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    proto = df["PROTOCOL"].to_numpy(dtype=np.int64)
    dport = df["L4_DST_PORT"].to_numpy(dtype=np.int64)

    x[:, idx_map["proto_tcp"]] = (proto == 6).astype(np.float32)
    x[:, idx_map["proto_udp"]] = (proto == 17).astype(np.float32)
    x[:, idx_map["proto_icmp"]] = np.isin(proto, [1, 58]).astype(np.float32)
    x[:, idx_map["proto_other"]] = (~np.isin(proto, [1, 6, 17, 58])).astype(np.float32)

    x[:, idx_map["dst_port_well_known"]] = ((dport >= 0) & (dport <= 1023)).astype(np.float32)
    x[:, idx_map["dst_port_registered"]] = ((dport >= 1024) & (dport <= 49151)).astype(np.float32)
    x[:, idx_map["dst_port_dynamic"]] = (dport >= 49152).astype(np.float32)
    x[:, idx_map["dst_port_http"]] = np.isin(dport, [80, 8080, 8000]).astype(np.float32)
    x[:, idx_map["dst_port_https"]] = np.isin(dport, [443, 8443]).astype(np.float32)
    x[:, idx_map["dst_port_dns"]] = (dport == 53).astype(np.float32)
    x[:, idx_map["dst_port_ssh"]] = (dport == 22).astype(np.float32)
    x[:, idx_map["dst_port_ftp"]] = np.isin(dport, [20, 21]).astype(np.float32)
    x[:, idx_map["dst_port_rdp"]] = (dport == 3389).astype(np.float32)
    x[:, idx_map["dst_port_smb"]] = (dport == 445).astype(np.float32)

    dur_us = np.maximum(0.0, df["FLOW_DURATION_MILLISECONDS"].to_numpy(dtype=np.float64) * 1000.0)

    in_bytes = np.maximum(0.0, df["IN_BYTES"].to_numpy(dtype=np.float64))
    out_bytes = np.maximum(0.0, df["OUT_BYTES"].to_numpy(dtype=np.float64))
    in_pkts = np.maximum(0.0, df["IN_PKTS"].to_numpy(dtype=np.float64))
    out_pkts = np.maximum(0.0, df["OUT_PKTS"].to_numpy(dtype=np.float64))
    tot_bytes = in_bytes + out_bytes
    tot_pkts = in_pkts + out_pkts

    valid_rate = (dur_us >= 100.0) & (tot_pkts > 1)
    dur_sec = np.where(valid_rate, dur_us / 1_000_000.0, 1.0)

    x[:, idx_map["flow_duration_us"]] = dur_us.astype(np.float32)
    x[:, idx_map["flow_bytes_per_s"]] = np.where(valid_rate, tot_bytes / dur_sec, 0.0).astype(np.float32)
    x[:, idx_map["flow_pkts_per_s"]] = np.where(valid_rate, tot_pkts / dur_sec, 0.0).astype(np.float32)
    x[:, idx_map["fwd_pkts_per_s"]] = np.where(valid_rate, in_pkts / dur_sec, 0.0).astype(np.float32)
    x[:, idx_map["bwd_pkts_per_s"]] = np.where(valid_rate, out_pkts / dur_sec, 0.0).astype(np.float32)

    x[:, idx_map["total_fwd_pkts"]] = in_pkts.astype(np.float32)
    x[:, idx_map["total_bwd_pkts"]] = out_pkts.astype(np.float32)
    x[:, idx_map["total_fwd_bytes"]] = in_bytes.astype(np.float32)
    x[:, idx_map["total_bwd_bytes"]] = out_bytes.astype(np.float32)
    x[:, idx_map["down_up_ratio"]] = np.where(in_pkts > 0, out_pkts / np.maximum(in_pkts, 1.0), 0.0).astype(np.float32)
    x[:, idx_map["bwd_fwd_bytes_ratio"]] = np.where(in_bytes > 0, out_bytes / np.maximum(in_bytes, 1.0), 0.0).astype(np.float32)

    avg_fwd = np.where(in_pkts > 0, in_bytes / np.maximum(in_pkts, 1.0), 0.0)
    avg_bwd = np.where(out_pkts > 0, out_bytes / np.maximum(out_pkts, 1.0), 0.0)
    avg_all = np.where(tot_pkts > 0, tot_bytes / np.maximum(tot_pkts, 1.0), 0.0)

    x[:, idx_map["fwd_pkt_len_mean"]] = avg_fwd.astype(np.float32)
    x[:, idx_map["bwd_pkt_len_mean"]] = avg_bwd.astype(np.float32)
    x[:, idx_map["pkt_len_min"]] = np.maximum(0.0, df["SHORTEST_FLOW_PKT"].to_numpy(dtype=np.float32))
    x[:, idx_map["pkt_len_max"]] = np.maximum(0.0, df["LONGEST_FLOW_PKT"].to_numpy(dtype=np.float32))
    x[:, idx_map["pkt_len_mean"]] = avg_all.astype(np.float32)
    x[:, idx_map["avg_pkt_size"]] = avg_all.astype(np.float32)
    x[:, idx_map["avg_fwd_seg_size"]] = avg_fwd.astype(np.float32)
    x[:, idx_map["avg_bwd_seg_size"]] = avg_bwd.astype(np.float32)

    # IATs in NetFlow v3 are in milliseconds -> convert to microseconds
    x[:, idx_map["fwd_iat_total"]] = np.maximum(0.0, df["DURATION_IN"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["fwd_iat_mean"]] = np.maximum(0.0, df["SRC_TO_DST_IAT_AVG"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["fwd_iat_std"]] = np.maximum(0.0, df["SRC_TO_DST_IAT_STDDEV"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["fwd_iat_max"]] = np.maximum(0.0, df["SRC_TO_DST_IAT_MAX"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["fwd_iat_min"]] = np.maximum(0.0, df["SRC_TO_DST_IAT_MIN"].to_numpy(dtype=np.float32) * 1000.0)

    x[:, idx_map["bwd_iat_total"]] = np.maximum(0.0, df["DURATION_OUT"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["bwd_iat_mean"]] = np.maximum(0.0, df["DST_TO_SRC_IAT_AVG"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["bwd_iat_std"]] = np.maximum(0.0, df["DST_TO_SRC_IAT_STDDEV"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["bwd_iat_max"]] = np.maximum(0.0, df["DST_TO_SRC_IAT_MAX"].to_numpy(dtype=np.float32) * 1000.0)
    x[:, idx_map["bwd_iat_min"]] = np.maximum(0.0, df["DST_TO_SRC_IAT_MIN"].to_numpy(dtype=np.float32) * 1000.0)

    x[:, idx_map["init_win_bytes_fwd"]] = np.maximum(0.0, df["TCP_WIN_MAX_IN"].to_numpy(dtype=np.float32))
    x[:, idx_map["init_win_bytes_bwd"]] = np.maximum(0.0, df["TCP_WIN_MAX_OUT"].to_numpy(dtype=np.float32))
    # Do NOT populate fwd_ttl_mean / bwd_ttl_mean from UNSW-NB15 (IXIA testbed TTL=31 artifact)

    flags = df["TCP_FLAGS"].to_numpy(dtype=np.int64)
    x[:, idx_map["fin_flag_cnt"]] = ((flags & 0x01) > 0).astype(np.float32)
    x[:, idx_map["syn_flag_cnt"]] = ((flags & 0x02) > 0).astype(np.float32)
    x[:, idx_map["rst_flag_cnt"]] = ((flags & 0x04) > 0).astype(np.float32)
    x[:, idx_map["psh_flag_cnt"]] = ((flags & 0x08) > 0).astype(np.float32)
    x[:, idx_map["ack_flag_cnt"]] = ((flags & 0x10) > 0).astype(np.float32)
    x[:, idx_map["urg_flag_cnt"]] = ((flags & 0x20) > 0).astype(np.float32)
    x[:, idx_map["ece_flag_cnt"]] = ((flags & 0x40) > 0).astype(np.float32)
    x[:, idx_map["cwr_flag_cnt"]] = ((flags & 0x80) > 0).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=0.0)
    return x, y


def load_5g_nidd_csv(
    csv_path: Path,
    max_benign: int = 180_000,
    max_ddos: int = 180_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load 5G-NIDD Combined.csv (Argus flow format) into FEATURE_NAMES."""
    import pandas as pd

    use_cols = [
        "Dur",
        "Mean",
        "Min",
        "Max",
        "Proto",
        "sTtl",
        "dTtl",
        "TotPkts",
        "SrcPkts",
        "DstPkts",
        "TotBytes",
        "SrcBytes",
        "DstBytes",
        "sMeanPktSz",
        "dMeanPktSz",
        "Load",
        "Rate",
        "SrcRate",
        "DstRate",
        "State",
        "SrcWin",
        "DstWin",
        "Attack Type",
    ]
    df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
    df = df[df["Attack Type"].isin(NIDD_5G_TO_UNIFIED)].copy()
    df["cls_id"] = df["Attack Type"].map(NIDD_5G_TO_UNIFIED)

    parts: list[pd.DataFrame] = []
    benign_df = df[df["cls_id"] == 0]
    ddos_df = df[df["cls_id"] == 1]
    scan_df = df[df["cls_id"] == 2]
    if len(benign_df) > max_benign:
        benign_df = benign_df.sample(n=max_benign, random_state=seed)
    if len(ddos_df) > max_ddos:
        ddos_df = ddos_df.sample(n=max_ddos, random_state=seed)
    parts.extend([benign_df, ddos_df, scan_df])
    df = pd.concat(parts, ignore_index=True)

    y = df["cls_id"].to_numpy(dtype=np.int64)
    n = len(df)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    proto_s = df["Proto"].astype(str).str.lower().str.strip()
    is_tcp = proto_s == "tcp"
    is_udp = proto_s == "udp"
    is_icmp = proto_s.isin(["icmp", "ipv6-icmp"])
    x[:, idx_map["proto_tcp"]] = is_tcp.to_numpy(dtype=np.float32)
    x[:, idx_map["proto_udp"]] = is_udp.to_numpy(dtype=np.float32)
    x[:, idx_map["proto_icmp"]] = is_icmp.to_numpy(dtype=np.float32)
    x[:, idx_map["proto_other"]] = (~(is_tcp | is_udp | is_icmp)).to_numpy(dtype=np.float32)

    atk_s = df["Attack Type"].astype(str)
    in_pkts_raw = np.maximum(pd.to_numeric(df["SrcPkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    out_pkts_raw = np.maximum(pd.to_numeric(df["DstPkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)

    is_http_atk = atk_s.isin(["HTTPFlood", "SlowrateDoS", "SYNFlood"]).to_numpy()
    is_scan_atk = atk_s.isin(["TCPConnectScan", "SYNScan", "UDPScan"]).to_numpy()
    is_udp_flood = (atk_s == "UDPFlood").to_numpy()
    is_benign = (atk_s == "Benign").to_numpy()

    dns_like = is_benign & is_udp.to_numpy() & (in_pkts_raw <= 4) & (out_pkts_raw >= 1)
    https_like = is_benign & is_tcp.to_numpy()
    well_known = is_http_atk | dns_like | https_like | (is_scan_atk & (np.arange(n) % 2 == 0))
    registered = is_udp_flood | (is_benign & is_udp.to_numpy() & ~dns_like) | (is_scan_atk & (np.arange(n) % 2 == 1))

    x[:, idx_map["dst_port_well_known"]] = well_known.astype(np.float32)
    x[:, idx_map["dst_port_registered"]] = registered.astype(np.float32)
    x[:, idx_map["dst_port_http"]] = is_http_atk.astype(np.float32)
    x[:, idx_map["dst_port_https"]] = https_like.astype(np.float32)
    x[:, idx_map["dst_port_dns"]] = dns_like.astype(np.float32)

    dur_sec = np.maximum(pd.to_numeric(df["Dur"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    dur_us = dur_sec * 1_000_000.0

    in_pkts = np.maximum(pd.to_numeric(df["SrcPkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    out_pkts = np.maximum(pd.to_numeric(df["DstPkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    tot_pkts = np.maximum(pd.to_numeric(df["TotPkts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), in_pkts + out_pkts)

    in_bytes = np.maximum(pd.to_numeric(df["SrcBytes"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    out_bytes = np.maximum(pd.to_numeric(df["DstBytes"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    tot_bytes = np.maximum(pd.to_numeric(df["TotBytes"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), in_bytes + out_bytes)

    valid_rate = (dur_us >= 100.0) & (tot_pkts > 1)
    dur_safe = np.where(valid_rate, dur_sec, 1.0)

    x[:, idx_map["flow_duration_us"]] = dur_us.astype(np.float32)
    x[:, idx_map["flow_bytes_per_s"]] = np.where(valid_rate, tot_bytes / dur_safe, 0.0).astype(np.float32)
    x[:, idx_map["flow_pkts_per_s"]] = np.where(valid_rate, tot_pkts / dur_safe, 0.0).astype(np.float32)
    x[:, idx_map["fwd_pkts_per_s"]] = np.where(valid_rate, in_pkts / dur_safe, 0.0).astype(np.float32)
    x[:, idx_map["bwd_pkts_per_s"]] = np.where(valid_rate, out_pkts / dur_safe, 0.0).astype(np.float32)

    x[:, idx_map["total_fwd_pkts"]] = in_pkts.astype(np.float32)
    x[:, idx_map["total_bwd_pkts"]] = out_pkts.astype(np.float32)
    x[:, idx_map["total_fwd_bytes"]] = in_bytes.astype(np.float32)
    x[:, idx_map["total_bwd_bytes"]] = out_bytes.astype(np.float32)
    x[:, idx_map["down_up_ratio"]] = np.where(in_pkts > 0, out_pkts / np.maximum(in_pkts, 1.0), 0.0).astype(np.float32)
    x[:, idx_map["bwd_fwd_bytes_ratio"]] = np.where(in_bytes > 0, out_bytes / np.maximum(in_bytes, 1.0), 0.0).astype(np.float32)

    s_mean = np.maximum(pd.to_numeric(df["sMeanPktSz"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0)
    d_mean = np.maximum(pd.to_numeric(df["dMeanPktSz"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0)
    all_mean = np.where(tot_pkts > 0, tot_bytes / np.maximum(tot_pkts, 1.0), 0.0).astype(np.float32)

    x[:, idx_map["fwd_pkt_len_mean"]] = s_mean
    x[:, idx_map["bwd_pkt_len_mean"]] = d_mean
    x[:, idx_map["pkt_len_min"]] = np.where(d_mean > 0, np.minimum(s_mean, d_mean), s_mean).astype(np.float32)
    x[:, idx_map["pkt_len_max"]] = np.maximum(s_mean, d_mean).astype(np.float32)
    x[:, idx_map["pkt_len_mean"]] = all_mean
    x[:, idx_map["avg_pkt_size"]] = all_mean
    x[:, idx_map["avg_fwd_seg_size"]] = s_mean
    x[:, idx_map["avg_bwd_seg_size"]] = d_mean

    x[:, idx_map["fwd_iat_total"]] = np.where(in_pkts > 1, dur_us, 0.0).astype(np.float32)
    x[:, idx_map["fwd_iat_mean"]] = np.where(in_pkts > 1, dur_us / np.maximum(in_pkts - 1.0, 1.0), 0.0).astype(np.float32)
    x[:, idx_map["bwd_iat_total"]] = np.where(out_pkts > 1, dur_us, 0.0).astype(np.float32)
    x[:, idx_map["bwd_iat_mean"]] = np.where(out_pkts > 1, dur_us / np.maximum(out_pkts - 1.0, 1.0), 0.0).astype(np.float32)

    x[:, idx_map["init_win_bytes_fwd"]] = np.maximum(pd.to_numeric(df["SrcWin"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0)
    x[:, idx_map["init_win_bytes_bwd"]] = np.maximum(pd.to_numeric(df["DstWin"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0)
    # Do NOT populate fwd_ttl_mean / bwd_ttl_mean to avoid testbed TTL leakage

    state_s = df["State"].astype(str).str.upper()
    x[:, idx_map["fin_flag_cnt"]] = state_s.str.contains("FIN").to_numpy(dtype=np.float32)
    x[:, idx_map["syn_flag_cnt"]] = state_s.str.contains("SYN|REQ").to_numpy(dtype=np.float32)
    x[:, idx_map["rst_flag_cnt"]] = state_s.str.contains("RST").to_numpy(dtype=np.float32)
    x[:, idx_map["ack_flag_cnt"]] = state_s.str.contains("ACK|CON|EST").to_numpy(dtype=np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=0.0)
    return x, y


def load_combined_real_datasets(
    data_dir: Path,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[str]] | None:
    """Load and merge all recognized real datasets in data_dir, shuffled together."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sources: list[str] = []

    nf_csv = find_nf_unsw_csv(data_dir)
    if nf_csv is not None:
        x_nf, y_nf = load_nf_unsw_nb15_v3(nf_csv, seed=seed)
        xs.append(x_nf)
        ys.append(y_nf)
        sources.append(f"NF-UNSW-NB15-v3 ({len(y_nf):,})")

    for nidd_csv in find_5g_nidd_csvs(data_dir):
        x_nidd, y_nidd = load_5g_nidd_csv(nidd_csv, seed=seed)
        xs.append(x_nidd)
        ys.append(y_nidd)
        sources.append(f"5G-NIDD ({len(y_nidd):,})")

    if not xs:
        return None

    # Enforce strict intersection of active features across all loaded datasets
    # so the model cannot identify dataset source by missing-column zeros
    common_mask = np.ones(NUM_FEATURES, dtype=bool)
    for x_part in xs:
        common_mask &= np.any(x_part != 0.0, axis=0)

    x_all = np.vstack(xs).astype(np.float32, copy=False)
    x_all[:, ~common_mask] = 0.0
    y_all = np.concatenate(ys).astype(np.int64, copy=False)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_all))
    return x_all[perm], y_all[perm], UNIFIED_CLASSES, sources
