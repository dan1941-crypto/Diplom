"""Background QThread workers for PCAP, Live Npcap, and Autonomous Demo streaming."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from src.capture.flow_table import FlowRecord, FlowTable
from src.capture.packet_parser import DLT_EN10MB, DLT_NULL, iter_pcap_packets, parse_raw_packet
from src.config import AppConfig
from src.ml.dataset import _simulate_flow
from src.ml.inference import FlowInferenceEngine, FlowPrediction


def list_network_interfaces() -> list[tuple[str, str]]:
    """Return list of (iface_id, display_name) tuples safely."""
    try:
        from scapy.arch.windows import get_windows_if_list  # type: ignore

        ifaces = get_windows_if_list()
        result: list[tuple[str, str]] = []
        for item in ifaces:
            name = item.get("name") or item.get("description") or ""
            desc = item.get("description") or name
            guid = item.get("guid") or name
            if name:
                result.append((name, f"{desc} ({name})"))
        if result:
            return result
    except Exception:
        pass

    try:
        from scapy.all import get_if_list  # type: ignore

        return [(iface, iface) for iface in get_if_list()]
    except Exception:
        return []


class CaptureInferenceWorker(QObject):
    """Runs packet capture / flow aggregation / ONNX inference off the main GUI thread."""

    batch_ready = Signal(list, dict)  # (list[FlowPrediction], stats_dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        cfg: AppConfig,
        engine: FlowInferenceEngine,
        mode: str = "pcap",
        pcap_path: str | Path | None = None,
        interface: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.engine = engine
        self.mode = mode
        self.pcap_path = Path(pcap_path) if pcap_path else cfg.default_pcap
        self.interface = interface
        self._stop_requested = False

    @Slot()
    def request_stop(self) -> None:
        """Signal the worker loop to stop cleanly."""
        self._stop_requested = True

    @Slot()
    def run(self) -> None:
        """Execute the selected capture & inference pipeline."""
        try:
            if self.mode == "pcap":
                self._run_pcap()
            elif self.mode == "live":
                self._run_live()
            elif self.mode == "demo":
                self._run_demo()
            else:
                self.error_occurred.emit(f"Unknown capture mode: {self.mode}")
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def _emit_batch(self, flows: list[FlowRecord], pkt_count: int, active_flows: int) -> None:
        if not flows:
            self.batch_ready.emit(
                [],
                {"packets": pkt_count, "active_flows": active_flows},
            )
            return
        preds = self.engine.predict_batch(flows)
        self.batch_ready.emit(
            preds,
            {"packets": pkt_count, "active_flows": active_flows},
        )

    def _run_pcap(self) -> None:
        if not self.pcap_path.exists():
            self.error_occurred.emit(f"PCAP-файл не найден: {self.pcap_path}")
            return

        self.status_changed.emit(f"Анализ PCAP: {self.pcap_path.name}")
        table = FlowTable(
            idle_timeout_sec=self.cfg.idle_timeout_sec,
            active_timeout_sec=self.cfg.active_timeout_sec,
            max_flows=self.cfg.max_flows,
        )

        pending_flows: list[FlowRecord] = []
        pkt_count = 0
        last_ts = 0.0
        batch_size = self.cfg.capture_batch_size

        for pkt in iter_pcap_packets(self.pcap_path):
            if self._stop_requested:
                break
            pkt_count += 1
            last_ts = pkt.timestamp
            completed = table.process_packet(pkt)
            if completed:
                pending_flows.extend(completed)

            if pkt_count % 2000 == 0 and last_ts > 0:
                pending_flows.extend(table.flush_expired(last_ts))

            if len(pending_flows) >= batch_size:
                self._emit_batch(pending_flows, pkt_count, table.active_count)
                pending_flows = []

        if not self._stop_requested:
            pending_flows.extend(table.flush_all())
        if pending_flows:
            self._emit_batch(pending_flows, pkt_count, table.active_count)
        else:
            self._emit_batch([], pkt_count, table.active_count)

        self.status_changed.emit(f"Анализ PCAP завершён ({pkt_count:,} пакетов)")

    def _run_live(self) -> None:
        try:
            from scapy.all import AsyncSniffer  # type: ignore
        except ImportError as exc:
            self.error_occurred.emit(f"Scapy is required for live capture: {exc}")
            return

        pkt_queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue(maxsize=20_000)

        def _on_packet(pkt: Any) -> None:
            try:
                dlt = DLT_NULL if pkt.__class__.__name__ == "Loopback" else DLT_EN10MB
                pkt_queue.put_nowait((bytes(pkt), float(pkt.time), dlt))
            except queue.Full:
                pass

        sniffers: list[Any] = []
        active_names: list[str] = []

        if self.interface == "__ALL__":
            candidates = [iface_id for iface_id, _ in list_network_interfaces() if iface_id]
            for iface_id in candidates:
                try:
                    sn = AsyncSniffer(iface=iface_id, store=False, prn=_on_packet)
                    sn.start()
                    sniffers.append(sn)
                    active_names.append(iface_id)
                except Exception:
                    continue
            if not sniffers:
                self.error_occurred.emit(
                    "Не удалось открыть ни один сетевой адаптер. Убедитесь, что в Windows установлен Npcap "
                    "и есть права администратора на захват трафика."
                )
                return
            self.status_changed.emit(f"Живой захват по всем адаптерам ({len(active_names)} активных)")
        else:
            sniffer_kwargs: dict[str, Any] = {"store": False, "prn": _on_packet}
            if self.interface:
                sniffer_kwargs["iface"] = self.interface
            try:
                sn = AsyncSniffer(**sniffer_kwargs)
                sn.start()
                sniffers.append(sn)
            except Exception as exc:
                self.error_occurred.emit(
                    "Не удалось запустить живой захват. Убедитесь, что в Windows установлен драйвер Npcap "
                    f"и есть права администратора на захват трафика.\nДетали: {exc}"
                )
                return
            self.status_changed.emit(f"Живой захват на: {self.interface or 'интерфейсе по умолчанию'}")

        table = FlowTable(
            idle_timeout_sec=self.cfg.idle_timeout_sec,
            active_timeout_sec=self.cfg.active_timeout_sec,
            max_flows=self.cfg.max_flows,
        )
        pkt_count = 0
        pending_flows: list[FlowRecord] = []
        last_flush = time.monotonic()
        interval_sec = self.cfg.capture_batch_interval_ms / 1000.0

        try:
            while not self._stop_requested:
                try:
                    raw_bytes, ts, dlt = pkt_queue.get(timeout=0.05)
                    parsed = parse_raw_packet(raw_bytes, ts, datalink=dlt)
                    if parsed is not None:
                        pkt_count += 1
                        completed = table.process_packet(parsed)
                        if completed:
                            pending_flows.extend(completed)
                except queue.Empty:
                    pass

                now_mono = time.monotonic()
                if now_mono - last_flush >= interval_sec:
                    pending_flows.extend(table.flush_expired(time.time()))
                    self._emit_batch(pending_flows, pkt_count, table.active_count)
                    pending_flows = []
                    last_flush = now_mono
        finally:
            for sn in sniffers:
                try:
                    sn.stop()
                except Exception:
                    pass
            pending_flows.extend(table.flush_all())
            if pending_flows:
                self._emit_batch(pending_flows, pkt_count, 0)
            self.status_changed.emit("Живой захват остановлен")

    def _run_demo(self) -> None:
        """Autonomous real-time stream simulator for testing without Npcap."""
        self.status_changed.emit("Автономный демо-поток активен")
        rng = np.random.default_rng()
        pkt_count = 0
        probs = np.array([0.70, 0.12, 0.08, 0.06, 0.04], dtype=np.float64)

        while not self._stop_requested:
            batch_n = int(rng.integers(6, 18))
            now_ts = time.time()
            flows: list[FlowRecord] = []
            for i in range(batch_n):
                cls_id = int(rng.choice(5, p=probs))
                flow = _simulate_flow(rng, cls_id, now_ts + i * 0.005)
                pkt_count += len(flow.all_lengths)
                flows.append(flow)

            self._emit_batch(flows, pkt_count, int(rng.integers(12, 48)))
            time.sleep(0.35)

        self.status_changed.emit("Демо-поток остановлен")
