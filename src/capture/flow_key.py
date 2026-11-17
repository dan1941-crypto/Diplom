"""Symmetric bidirectional flow key normalization."""

from __future__ import annotations

from typing import NamedTuple


class FlowKey(NamedTuple):
    """Canonical bidirectional 5-tuple flow identifier."""

    ep_a_ip: str
    ep_a_port: int
    ep_b_ip: str
    ep_b_port: int
    proto: int


def make_flow_key(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    proto: int,
) -> FlowKey:
    """Build a symmetric FlowKey so forward and backward packets share one flow.

    Endpoints are ordered lexicographically by ``(ip, port)``.
    """
    ep1 = (src_ip, int(src_port))
    ep2 = (dst_ip, int(dst_port))
    if ep1 <= ep2:
        return FlowKey(ep1[0], ep1[1], ep2[0], ep2[1], int(proto))
    return FlowKey(ep2[0], ep2[1], ep1[0], ep1[1], int(proto))
