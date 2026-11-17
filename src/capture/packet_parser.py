"""Fast packet decoding for PCAP, PCAPNG, and live raw frames using dpkt."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import dpkt

# Standard DLT constants
DLT_NULL = 0
DLT_EN10MB = 1
DLT_RAW = 101
DLT_LINUX_SLL = 113

# TCP flag bitmasks
TH_FIN = 0x01
TH_SYN = 0x02
TH_RST = 0x04
TH_PSH = 0x08
TH_ACK = 0x10
TH_URG = 0x20
TH_ECE = 0x40
TH_CWR = 0x80


@dataclass(slots=True)
class ParsedPacket:
    """Normalized L3/L4 packet metadata extracted from wire bytes."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    ip_len: int
    payload_len: int
    header_len: int
    tcp_flags: int
    tcp_window: int
    ttl: int


def _inet_to_str(inet: bytes) -> str:
    """Convert raw IPv4/IPv6 address bytes to string."""
    if len(inet) == 4:
        return socket.inet_ntop(socket.AF_INET, inet)
    if len(inet) == 16:
        return socket.inet_ntop(socket.AF_INET6, inet)
    return ""


def _decode_ip_packet(ip: dpkt.ip.IP | dpkt.ip6.IP6, timestamp: float) -> ParsedPacket | None:
    """Extract L3/L4 fields from a dpkt IPv4 or IPv6 object."""
    if isinstance(ip, dpkt.ip.IP):
        src_ip = _inet_to_str(ip.src)
        dst_ip = _inet_to_str(ip.dst)
        proto = int(ip.p)
        ip_len = int(ip.len) if ip.len else len(bytes(ip))
        ttl = int(ip.ttl)
        l4 = ip.data
    elif isinstance(ip, dpkt.ip6.IP6):
        src_ip = _inet_to_str(ip.src)
        dst_ip = _inet_to_str(ip.dst)
        proto = int(ip.nxt)
        ip_len = int(ip.plen) + 40
        ttl = int(ip.hlim)
        l4 = ip.data
    else:
        return None

    if not src_ip or not dst_ip:
        return None

    src_port = 0
    dst_port = 0
    header_len = 0
    payload_len = 0
    tcp_flags = 0
    tcp_window = 0

    if isinstance(l4, dpkt.tcp.TCP):
        src_port = int(l4.sport)
        dst_port = int(l4.dport)
        header_len = max(20, int(l4.off) * 4)
        payload_len = len(l4.data) if isinstance(l4.data, (bytes, bytearray)) else 0
        tcp_flags = int(l4.flags)
        tcp_window = int(l4.win)
    elif isinstance(l4, dpkt.udp.UDP):
        src_port = int(l4.sport)
        dst_port = int(l4.dport)
        header_len = 8
        payload_len = len(l4.data) if isinstance(l4.data, (bytes, bytearray)) else max(0, int(l4.ulen) - 8)
    elif isinstance(l4, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
        header_len = 8
        payload_len = len(l4.data) if isinstance(l4.data, (bytes, bytearray)) else 0
    else:
        payload_len = len(l4) if isinstance(l4, (bytes, bytearray)) else 0

    return ParsedPacket(
        timestamp=float(timestamp),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        proto=proto,
        ip_len=max(0, ip_len),
        payload_len=max(0, payload_len),
        header_len=max(0, header_len),
        tcp_flags=tcp_flags,
        tcp_window=tcp_window,
        ttl=ttl,
    )


def parse_raw_packet(raw_bytes: bytes, timestamp: float, datalink: int = DLT_EN10MB) -> ParsedPacket | None:
    """Decode a single raw frame into a ParsedPacket, or None if non-IP/malformed."""
    try:
        if datalink == DLT_NULL:
            if len(raw_bytes) > 4:
                ver = raw_bytes[4] >> 4
                if ver == 4:
                    return _decode_ip_packet(dpkt.ip.IP(raw_bytes[4:]), timestamp)
                if ver == 6:
                    return _decode_ip_packet(dpkt.ip6.IP6(raw_bytes[4:]), timestamp)
            return None
        if datalink == DLT_EN10MB:
            eth = dpkt.ethernet.Ethernet(raw_bytes)
            if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return _decode_ip_packet(eth.data, timestamp)
            return None
        if datalink == DLT_LINUX_SLL:
            sll = dpkt.sll.SLL(raw_bytes)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return _decode_ip_packet(sll.data, timestamp)
            return None
        if datalink == DLT_RAW:
            ver = (raw_bytes[0] >> 4) if raw_bytes else 0
            if ver == 4:
                return _decode_ip_packet(dpkt.ip.IP(raw_bytes), timestamp)
            if ver == 6:
                return _decode_ip_packet(dpkt.ip6.IP6(raw_bytes), timestamp)
            return None
        # Fallback: try Ethernet first
        eth = dpkt.ethernet.Ethernet(raw_bytes)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return _decode_ip_packet(eth.data, timestamp)
    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, IndexError, ValueError):
        return None
    return None


def iter_pcap_packets(pcap_path: str | Path) -> Iterator[ParsedPacket]:
    """Stream ParsedPacket objects from a .pcap or .pcapng file."""
    path = Path(pcap_path)
    with path.open("rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            reader = dpkt.pcapng.Reader(f)
        else:
            reader = dpkt.pcap.Reader(f)

        datalink = getattr(reader, "datalink", lambda: DLT_EN10MB)()
        for ts, buf in reader:
            pkt = parse_raw_packet(buf, ts, datalink=datalink)
            if pkt is not None:
                yield pkt
