"""PySide6 main window for NIDS Traffic Analyzer."""

from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.ml.inference import FlowInferenceEngine, FlowPrediction
from src.ui.workers import CaptureInferenceWorker, list_network_interfaces

PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP", 58: "ICMPv6"}
SEVERITY_RU = {
    "Normal": "Норма",
    "Suspicious": "Подозрительно",
    "Attack": "Атака",
}


def _format_ts(ts: float, iso: bool = False) -> str:
    """Safely format epoch or relative PCAPNG timestamps on Windows."""
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.isoformat() if iso else dt.strftime("%H:%M:%S.%f")[:-3]
    except (OSError, OverflowError, ValueError):
        return f"{ts:.3f}s"

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QFrame#KpiCard {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
}
QFrame#Panel {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #334155; color: #64748b; }
QPushButton#StopBtn { background-color: #dc2626; }
QPushButton#StopBtn:hover { background-color: #b91c1c; }
QPushButton#SecondaryBtn { background-color: #334155; color: #e2e8f0; }
QPushButton#SecondaryBtn:hover { background-color: #475569; }
QComboBox, QLineEdit {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 8px;
    color: #f8fafc;
}
QTableWidget {
    background-color: #0f172a;
    alternate-background-color: #162033;
    gridline-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    selection-background-color: #1d4ed8;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #334155;
    font-weight: 600;
}
"""


class _TrainWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @Slot()
    def run(self) -> None:
        try:
            from src.ml.train import train_and_export

            meta = train_and_export(self.cfg, progress_callback=self.progress.emit)
            self.finished.emit(meta)
        except ImportError:
            self.failed.emit(
                "Модуль обучения (PyTorch) не включён в автономный .exe. "
                "Для переобучения модели запустите проект из исходников (python -m src.ml.train)."
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingProgressDialog(QDialog):
    """Real-time training progress window with progress bar, metrics, and live log."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Обучение Tabular ResNet (Real-Time)")
        self.resize(680, 440)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.header_lbl = QLabel("Инициализация обучения...")
        self.header_lbl.setStyleSheet("font-weight: 700; font-size: 14px; color: #38bdf8;")
        layout.addWidget(self.header_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.metrics_lbl = QLabel("Ожидание запуска эпох...")
        self.metrics_lbl.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        layout.addWidget(self.metrics_lbl)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background-color: #0b1120; color: #cbd5e1; font-family: Consolas, monospace; font-size: 12px;"
        )
        layout.addWidget(self.log_view, 1)

        self.close_btn = QPushButton("Закрыть")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn, alignment=Qt.AlignmentFlag.AlignRight)

    @Slot(dict)
    def on_progress(self, ev: dict[str, Any]) -> None:
        stage = ev.get("stage", "")
        if stage == "data":
            msg = str(ev.get("message", ""))
            self.header_lbl.setText(f"Устройство: {ev.get('device', 'GPU/CPU')}")
            self.log_view.appendPlainText(msg)
        elif stage == "epoch":
            ep = int(ev.get("epoch", 0))
            tot = max(1, int(ev.get("epochs", 1)))
            loss = float(ev.get("loss", 0.0))
            val_f1 = float(ev.get("val_macro_f1", 0.0))
            elapsed = float(ev.get("elapsed_s", 0.0))
            pct = int((ep / tot) * 100)
            self.progress_bar.setValue(pct)
            self.metrics_lbl.setText(
                f"Эпоха {ep}/{tot} | Loss: {loss:.4f} | Val Macro-F1: {val_f1:.4f} | Время: {elapsed:.1f} с"
            )
            self.log_view.appendPlainText(
                f"Epoch {ep:02d}/{tot:02d} | loss={loss:.4f} | val_macro_f1={val_f1:.4f} | {elapsed:.1f}s"
            )
        elif stage == "eval":
            self.progress_bar.setValue(100)
            self.header_lbl.setText(str(ev.get("message", "Готово")))
            report = str(ev.get("report", ""))
            if report:
                self.log_view.appendPlainText("\n=== Test Classification Report ===\n" + report)
            self.close_btn.setEnabled(True)

    @Slot(str)
    def on_error(self, err: str) -> None:
        self.header_lbl.setText("Ошибка обучения")
        self.log_view.appendPlainText(f"\nERROR: {err}")
        self.close_btn.setEnabled(True)


class KpiCard(QFrame):
    """Compact metric card for the top dashboard row."""

    def __init__(self, title: str, accent: str = "#38bdf8") -> None:
        super().__init__()
        self.setObjectName("KpiCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        self.val_lbl = QLabel("0")
        self.val_lbl.setStyleSheet(f"color: {accent}; font-size: 20px; font-weight: 700;")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.val_lbl)

    def set_value(self, text: str) -> None:
        self.val_lbl.setText(text)


class MainWindow(QMainWindow):
    """Main desktop dashboard for live and PCAP intrusion detection."""

    def __init__(self, cfg: AppConfig, engine: FlowInferenceEngine) -> None:
        super().__init__()
        self.cfg = cfg
        self.engine = engine

        self.setWindowTitle("NIDS Traffic Analyzer — Анализатор сетевого трафика (Tabular ResNet + ONNX)")
        self.resize(1380, 860)

        self._worker_thread: QThread | None = None
        self._worker: CaptureInferenceWorker | None = None
        self._train_thread: QThread | None = None
        self._train_worker: _TrainWorker | None = None
        self._train_dialog: TrainingProgressDialog | None = None

        self._predictions: list[FlowPrediction] = []
        self._total_flows = 0
        self._normal_count = 0
        self._suspicious_count = 0
        self._attack_count = 0
        self._packet_count = 0
        self._latency_sum = 0.0

        self._chart_x: deque[int] = deque(maxlen=120)
        self._chart_flows: deque[int] = deque(maxlen=120)
        self._chart_threats: deque[int] = deque(maxlen=120)
        self._tick = 0

        self._build_ui()
        self._refresh_model_badge()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 1. Top Control Bar
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("Panel")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(8)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("PCAP-файл", "pcap")
        self.mode_combo.addItem("Живой захват (Npcap)", "live")
        self.mode_combo.addItem("Автономный демо-поток", "demo")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.pcap_edit = QLineEdit(str(self.cfg.default_pcap))
        self.pcap_edit.setMinimumWidth(260)
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.setObjectName("SecondaryBtn")
        self.browse_btn.clicked.connect(self._browse_pcap)

        self.iface_combo = QComboBox()
        self.iface_combo.setMinimumWidth(220)
        self.iface_combo.addItem("Все адаптеры (All Interfaces)", "__ALL__")
        for iface_id, label in list_network_interfaces():
            self.iface_combo.addItem(label, iface_id)
        if self.iface_combo.count() == 1:
            self.iface_combo.addItem("Интерфейс по умолчанию", "")

        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self._start_capture)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_capture)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setObjectName("SecondaryBtn")
        self.clear_btn.clicked.connect(self._clear_results)

        self.retrain_btn = QPushButton("Переобучить модель")
        self.retrain_btn.setObjectName("SecondaryBtn")
        self.retrain_btn.clicked.connect(self._retrain_model)

        self.export_csv_btn = QPushButton("Экспорт CSV")
        self.export_csv_btn.setObjectName("SecondaryBtn")
        self.export_csv_btn.clicked.connect(self._export_csv)

        self.export_json_btn = QPushButton("Экспорт JSON")
        self.export_json_btn.setObjectName("SecondaryBtn")
        self.export_json_btn.clicked.connect(self._export_json)

        ctrl_layout.addWidget(QLabel("Режим:"))
        ctrl_layout.addWidget(self.mode_combo)
        ctrl_layout.addWidget(self.pcap_edit, 1)
        ctrl_layout.addWidget(self.browse_btn)
        ctrl_layout.addWidget(self.iface_combo)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.clear_btn)
        ctrl_layout.addWidget(self.retrain_btn)
        ctrl_layout.addWidget(self.export_csv_btn)
        ctrl_layout.addWidget(self.export_json_btn)
        root.addWidget(ctrl_frame)

        # 2. KPI Cards Row
        kpi_layout = QGridLayout()
        kpi_layout.setSpacing(8)
        self.kpi_flows = KpiCard("ВСЕГО ПОТОКОВ", "#38bdf8")
        self.kpi_pkts = KpiCard("ПАКЕТОВ ОБРАБОТАНО", "#a78bfa")
        self.kpi_normal = KpiCard("НОРМА", "#22c55e")
        self.kpi_suspicious = KpiCard("ПОДОЗРИТЕЛЬНО", "#f59e0b")
        self.kpi_attack = KpiCard("АТАКИ", "#ef4444")
        self.kpi_latency = KpiCard("СРЕДНЯЯ ЗАДЕРЖКА", "#e2e8f0")

        for idx, card in enumerate(
            [
                self.kpi_flows,
                self.kpi_pkts,
                self.kpi_normal,
                self.kpi_suspicious,
                self.kpi_attack,
                self.kpi_latency,
            ]
        ):
            kpi_layout.addWidget(card, 0, idx)
        root.addLayout(kpi_layout)

        # 3. Chart + Table + Inspector Splitter
        main_split = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # PyQtGraph Live Chart
        chart_frame = QFrame()
        chart_frame.setObjectName("Panel")
        chart_box = QVBoxLayout(chart_frame)
        chart_box.setContentsMargins(8, 8, 8, 8)
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget(title="Потоки и угрозы в реальном времени")
        self.plot_widget.setBackground("#0f172a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(180)
        self.plot_widget.addLegend()
        self.curve_flows = self.plot_widget.plot(
            pen=pg.mkPen("#38bdf8", width=2), name="Потоки / батч"
        )
        self.curve_threats = self.plot_widget.plot(
            pen=pg.mkPen("#ef4444", width=2), name="Угрозы / батч"
        )
        chart_box.addWidget(self.plot_widget)
        left_layout.addWidget(chart_frame, 1)

        # Filter bar + Events Table
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Фильтр угроз:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Все", "Подозрительные и атаки", "Только атаки"])
        self.filter_combo.currentIndexChanged.connect(self._rebuild_table)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        self.model_badge = QLabel("")
        self.model_badge.setStyleSheet("color: #94a3b8; font-size: 12px;")
        filter_row.addWidget(self.model_badge)
        left_layout.addLayout(filter_row)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Время",
                "Источник IP:Порт",
                "Назначение IP:Порт",
                "Протокол",
                "Пакеты (Пр/Обр)",
                "Байты",
                "Длительность",
                "Класс",
                "Уровень",
                "Уверенность",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        left_layout.addWidget(self.table, 2)

        main_split.addWidget(left_widget)

        # Right: Flow Inspection Panel
        insp_frame = QFrame()
        insp_frame.setObjectName("Panel")
        insp_frame.setMinimumWidth(340)
        insp_layout = QVBoxLayout(insp_frame)
        insp_layout.setContentsMargins(12, 12, 12, 12)
        insp_layout.setSpacing(8)

        insp_title = QLabel("Инспекция потока")
        insp_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        insp_layout.addWidget(insp_title)

        self.insp_summary = QLabel(
            "Выберите поток в таблице для просмотра вероятностей и ключевых признаков."
        )
        self.insp_summary.setWordWrap(True)
        self.insp_summary.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px;"
        )
        insp_layout.addWidget(self.insp_summary)

        insp_layout.addWidget(QLabel("Вероятности классов:"))
        self.probs_table = QTableWidget(0, 2)
        self.probs_table.setHorizontalHeaderLabels(["Класс", "Вероятность"])
        self.probs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.probs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.probs_table.setMaximumHeight(175)
        insp_layout.addWidget(self.probs_table)

        insp_layout.addWidget(QLabel("Ключевые признаки (отклонение):"))
        self.feat_table = QTableWidget(0, 3)
        self.feat_table.setHorizontalHeaderLabels(["Признак", "Значение", "|z|"])
        self.feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        insp_layout.addWidget(self.feat_table, 1)

        main_split.addWidget(insp_frame)
        main_split.setSizes([980, 380])
        root.addWidget(main_split, 1)

        self.statusBar().showMessage("Готов к работе")
        self._on_mode_changed()

    def _refresh_model_badge(self) -> None:
        metrics = self.engine.metadata.get("metrics", {})
        if metrics:
            self.model_badge.setText(
                f"ONNX ResNet | Macro-F1: {metrics.get('macro_f1', 0):.3f} | "
                f"PR-AUC: {metrics.get('macro_pr_auc', 0):.3f} | "
                f"FPR: {metrics.get('false_positive_rate', 0):.3f}"
            )
        else:
            self.model_badge.setText("ONNX ResNet загружен")

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        is_pcap = mode == "pcap"
        is_live = mode == "live"
        self.pcap_edit.setVisible(is_pcap)
        self.browse_btn.setVisible(is_pcap)
        self.iface_combo.setVisible(is_live)

    @Slot()
    def _browse_pcap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл PCAP / PCAPNG",
            str(self.cfg.default_pcap.parent),
            "Файлы захвата (*.pcap *.pcapng *.cap);;Все файлы (*)",
        )
        if path:
            self.pcap_edit.setText(path)

    @Slot()
    def _start_capture(self) -> None:
        if self._worker_thread is not None:
            return

        mode = self.mode_combo.currentData()
        pcap_path = self.pcap_edit.text().strip()
        iface = self.iface_combo.currentData()

        self._worker_thread = QThread(self)
        self._worker = CaptureInferenceWorker(
            cfg=self.cfg,
            engine=self.engine,
            mode=mode,
            pcap_path=pcap_path,
            interface=iface,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.status_changed.connect(self.statusBar().showMessage)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self._worker_thread.start()

    @Slot()
    def _stop_capture(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.statusBar().showMessage("Остановка...")

    @Slot(list, dict)
    def _on_batch_ready(self, preds: list[FlowPrediction], stats: dict[str, Any]) -> None:
        self._packet_count = int(stats.get("packets", self._packet_count))
        threats_in_batch = 0

        for p in preds:
            self._predictions.append(p)
            self._total_flows += 1
            self._latency_sum += p.latency_ms
            if p.severity == "Attack":
                self._attack_count += 1
                threats_in_batch += 1
            elif p.severity == "Suspicious":
                self._suspicious_count += 1
                threats_in_batch += 1
            else:
                self._normal_count += 1

            if self._matches_filter(p):
                self._append_table_row(p, len(self._predictions) - 1)

        self._tick += 1
        self._chart_x.append(self._tick)
        self._chart_flows.append(len(preds))
        self._chart_threats.append(threats_in_batch)
        self.curve_flows.setData(list(self._chart_x), list(self._chart_flows))
        self.curve_threats.setData(list(self._chart_x), list(self._chart_threats))

        self._update_kpis()

    def _matches_filter(self, p: FlowPrediction) -> bool:
        flt = self.filter_combo.currentText()
        if flt == "Только атаки":
            return p.severity == "Attack"
        if flt == "Подозрительные и атаки":
            return p.severity in ("Suspicious", "Attack")
        return True

    def _append_table_row(self, p: FlowPrediction, pred_idx: int) -> None:
        # Cap visible rows at 2000 for smooth UI responsiveness
        if self.table.rowCount() >= 2000:
            self.table.removeRow(0)

        row = self.table.rowCount()
        self.table.insertRow(row)

        f = p.flow
        ts_str = _format_ts(f.start_ts, iso=False)
        proto_str = PROTO_MAP.get(f.proto, str(f.proto))
        total_bytes = sum(f.all_lengths)
        dur_ms = max(0.0, (f.last_ts - f.start_ts) * 1000.0)
        dur_str = f"{dur_ms / 1000.0:.2f} с" if dur_ms >= 1000 else f"{dur_ms:.1f} мс"

        values = [
            ts_str,
            f"{f.src_ip}:{f.src_port}",
            f"{f.dst_ip}:{f.dst_port}",
            proto_str,
            f"{len(f.fwd_lengths)}/{len(f.bwd_lengths)}",
            f"{total_bytes:,}",
            dur_str,
            p.predicted_class,
            SEVERITY_RU.get(p.severity, p.severity),
            f"{p.confidence * 100:.1f}%",
        ]

        sev_color = {
            "Normal": QColor("#22c55e"),
            "Suspicious": QColor("#f59e0b"),
            "Attack": QColor("#ef4444"),
        }.get(p.severity, QColor("#e2e8f0"))

        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, pred_idx)
            if col in (7, 8):
                item.setForeground(sev_color)
                if p.severity == "Attack":
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
            self.table.setItem(row, col, item)

    @Slot()
    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        filtered = [
            (idx, p) for idx, p in enumerate(self._predictions) if self._matches_filter(p)
        ]
        for idx, p in filtered[-2000:]:
            self._append_table_row(p, idx)

    @Slot()
    def _on_row_selected(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        first_item = self.table.item(row, 0)
        if first_item is None:
            return
        pred_idx = first_item.data(Qt.ItemDataRole.UserRole)
        if pred_idx is None or not (0 <= pred_idx < len(self._predictions)):
            return

        p = self._predictions[pred_idx]
        f = p.flow
        proto_str = PROTO_MAP.get(f.proto, str(f.proto))
        self.insp_summary.setText(
            f"<b>{f.src_ip}:{f.src_port}</b> → <b>{f.dst_ip}:{f.dst_port}</b> ({proto_str})<br>"
            f"Класс: <b>{p.predicted_class}</b> | Уровень: <b>{SEVERITY_RU.get(p.severity, p.severity)}</b><br>"
            f"Риск атаки: {p.attack_score * 100:.1f}% | Пакеты (Пр/Обр): {len(f.fwd_lengths)}/{len(f.bwd_lengths)}"
        )

        self.probs_table.setRowCount(len(p.probabilities))
        for r, (cls_name, prob) in enumerate(p.probabilities.items()):
            self.probs_table.setItem(r, 0, QTableWidgetItem(cls_name))
            self.probs_table.setItem(r, 1, QTableWidgetItem(f"{prob * 100:.2f}%"))

        self.feat_table.setRowCount(len(p.top_features))
        for r, (fname, fval, zmag) in enumerate(p.top_features):
            self.feat_table.setItem(r, 0, QTableWidgetItem(fname))
            self.feat_table.setItem(r, 1, QTableWidgetItem(fval))
            self.feat_table.setItem(r, 2, QTableWidgetItem(f"{zmag:.2f}"))

    def _update_kpis(self) -> None:
        self.kpi_flows.set_value(f"{self._total_flows:,}")
        self.kpi_pkts.set_value(f"{self._packet_count:,}")
        self.kpi_normal.set_value(f"{self._normal_count:,}")
        self.kpi_suspicious.set_value(f"{self._suspicious_count:,}")
        self.kpi_attack.set_value(f"{self._attack_count:,}")
        avg_lat = (self._latency_sum / self._total_flows) if self._total_flows > 0 else 0.0
        self.kpi_latency.set_value(f"{avg_lat:.2f} мс")

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Ошибка захвата", msg)

    @Slot()
    def _on_worker_finished(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mode_combo.setEnabled(True)

    @Slot()
    def _clear_results(self) -> None:
        self._predictions.clear()
        self._total_flows = 0
        self._normal_count = 0
        self._suspicious_count = 0
        self._attack_count = 0
        self._packet_count = 0
        self._latency_sum = 0.0
        self._chart_x.clear()
        self._chart_flows.clear()
        self._chart_threats.clear()
        self._tick = 0
        self.curve_flows.setData([], [])
        self.curve_threats.setData([], [])
        self.table.setRowCount(0)
        self.probs_table.setRowCount(0)
        self.feat_table.setRowCount(0)
        self.insp_summary.setText(
            "Выберите поток в таблице для просмотра вероятностей и ключевых признаков."
        )
        self._update_kpis()

    @Slot()
    def _retrain_model(self) -> None:
        if self._train_thread is not None:
            return
        self.retrain_btn.setEnabled(False)
        self.statusBar().showMessage("Обучение Tabular ResNet и экспорт в ONNX...")
        self._train_dialog = TrainingProgressDialog(self)
        self._train_dialog.show()

        self._train_thread = QThread(self)
        self._train_worker = _TrainWorker(self.cfg)
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.progress.connect(self._train_dialog.on_progress)
        self._train_worker.finished.connect(self._on_retrain_finished)
        self._train_worker.failed.connect(self._on_retrain_failed)
        self._train_thread.start()

    @Slot(dict)
    def _on_retrain_finished(self, meta: dict[str, Any]) -> None:
        self._cleanup_train_thread()
        self.engine = FlowInferenceEngine(self.cfg)
        self._refresh_model_badge()
        m = meta.get("metrics", {})
        self.statusBar().showMessage(
            f"Модель переобучена (Macro-F1={m.get('macro_f1', 0):.4f}, FPR={m.get('false_positive_rate', 0):.4f})"
        )

    @Slot(str)
    def _on_retrain_failed(self, err: str) -> None:
        if self._train_dialog is not None:
            self._train_dialog.on_error(err)
        self._cleanup_train_thread()
        QMessageBox.critical(self, "Ошибка обучения", err)

    def _cleanup_train_thread(self) -> None:
        if self._train_thread is not None:
            self._train_thread.quit()
            self._train_thread.wait(3000)
            self._train_thread.deleteLater()
            self._train_thread = None
        if self._train_worker is not None:
            self._train_worker.deleteLater()
            self._train_worker = None
        self.retrain_btn.setEnabled(True)

    @Slot()
    def _export_csv(self) -> None:
        if not self._predictions:
            QMessageBox.information(self, "Экспорт CSV", "Нет проанализированных потоков для экспорта.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт потоков в CSV", "nids_events.csv", "CSV (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "src_ip",
                    "src_port",
                    "dst_ip",
                    "dst_port",
                    "proto",
                    "fwd_pkts",
                    "bwd_pkts",
                    "total_bytes",
                    "duration_ms",
                    "predicted_class",
                    "severity",
                    "confidence",
                    "attack_score",
                ]
            )
            for p in self._predictions:
                fl = p.flow
                writer.writerow(
                    [
                        _format_ts(fl.start_ts, iso=True),
                        fl.src_ip,
                        fl.src_port,
                        fl.dst_ip,
                        fl.dst_port,
                        PROTO_MAP.get(fl.proto, str(fl.proto)),
                        len(fl.fwd_lengths),
                        len(fl.bwd_lengths),
                        sum(fl.all_lengths),
                        round(max(0.0, (fl.last_ts - fl.start_ts) * 1000.0), 3),
                        p.predicted_class,
                        p.severity,
                        round(p.confidence, 4),
                        round(p.attack_score, 4),
                    ]
                )
        self.statusBar().showMessage(f"Экспортировано {len(self._predictions)} потоков в {path}")

    @Slot()
    def _export_json(self) -> None:
        if not self._predictions:
            QMessageBox.information(self, "Экспорт JSON", "Нет проанализированных потоков для экспорта.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт потоков в JSON", "nids_events.json", "JSON (*.json)")
        if not path:
            return
        payload = [
            {
                "timestamp": _format_ts(p.flow.start_ts, iso=True),
                "src_ip": p.flow.src_ip,
                "src_port": p.flow.src_port,
                "dst_ip": p.flow.dst_ip,
                "dst_port": p.flow.dst_port,
                "proto": PROTO_MAP.get(p.flow.proto, str(p.flow.proto)),
                "fwd_pkts": len(p.flow.fwd_lengths),
                "bwd_pkts": len(p.flow.bwd_lengths),
                "total_bytes": sum(p.flow.all_lengths),
                "predicted_class": p.predicted_class,
                "severity": p.severity,
                "confidence": round(p.confidence, 4),
                "attack_score": round(p.attack_score, 4),
                "probabilities": p.probabilities,
            }
            for p in self._predictions
        ]
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.statusBar().showMessage(f"Экспортировано {len(self._predictions)} потоков в {path}")

    def closeEvent(self, event: Any) -> None:
        self._stop_capture()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        super().closeEvent(event)


def run_gui(cfg: AppConfig, engine: FlowInferenceEngine) -> int:
    """Create QApplication, apply dark theme, and show MainWindow."""
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    win = MainWindow(cfg, engine)
    win.show()
    return app.exec()
