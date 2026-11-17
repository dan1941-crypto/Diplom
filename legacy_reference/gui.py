import sys
import os
import asyncio
from PySide6 import QtWidgets, QtCore, QtGui
from backend import analyze_pcap, format_flow_txt


class AnalyzerWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(dict)
    result_signal = QtCore.Signal(dict)
    error_signal = QtCore.Signal(str)

    def __init__(self, pcap_path: str):
        super().__init__()
        self.pcap_path = pcap_path

    def run(self):
        try:
            # Создаём event loop для pyshark в этом потоке
            try:
                asyncio.set_event_loop(asyncio.new_event_loop())
            except Exception:
                pass
            def cb(evt):
                self.progress_signal.emit(evt)
            res = analyze_pcap(self.pcap_path, progress_callback=cb)
            self.result_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


def apply_dark_theme(app: QtWidgets.QApplication):
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 38))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(76, 110, 245))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Анализатор трафика — AI')
        self.resize(1000, 700)
        # Шрифт: пробуем более выразительный, с запасным вариантом
        font = QtGui.QFont('Segoe UI Semibold', 10)
        if not QtGui.QFontInfo(font).family():
            font = QtGui.QFont('Segoe UI', 10)
        QtWidgets.QApplication.setFont(font)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # File chooser
        file_layout = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText('Выберите файл PCAP/PCAPNG...')
        browse_btn = QtWidgets.QPushButton('Обзор...')
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(browse_btn)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Анализ')
        self.start_btn.clicked.connect(self.start_analysis)
        self.export_all_btn = QtWidgets.QPushButton('Экспорт всех потоков')
        self.export_all_btn.clicked.connect(self.export_all)
        self.export_attacks_btn = QtWidgets.QPushButton('Экспорт атак')
        self.export_attacks_btn.clicked.connect(self.export_attacks)
        for b in (self.export_all_btn, self.export_attacks_btn):
            b.setEnabled(False)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(self.export_all_btn)
        ctrl_layout.addWidget(self.export_attacks_btn)

        # Progress + status
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.status_label = QtWidgets.QLabel('Готово')

        # Table of attacks
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['№', '5-tuple: src_ip, dst_ip, src_port, dst_port, protocol', 'Тип трафика'])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        # Подсказка по заголовку 5-tuple
        self.table.horizontalHeaderItem(1).setToolTip('Поток описывается как (src_ip, dst_ip, src_port, dst_port, proto)')

        layout.addLayout(file_layout)
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)

        self.worker: AnalyzerWorker | None = None
        self.last_result: dict | None = None

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Выберите PCAP/PCAPNG', '', 'PCAP Files (*.pcap *.pcapng);;All Files (*.*)')
        if path:
            self.path_edit.setText(path)

    def start_analysis(self):
        pcap_path = self.path_edit.text().strip()
        if not pcap_path or not os.path.exists(pcap_path):
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Укажите корректный файл PCAP/PCAPNG')
            return
        self.start_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)
        self.export_attacks_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText('Инициализация...')
        self.table.setRowCount(0)
        self.last_result = None

        self.worker = AnalyzerWorker(pcap_path)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.result_signal.connect(self.on_result)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_progress(self, evt: dict):
        stage = evt.get('stage')
        if stage == 'parsing':
            self.status_label.setText(f"Чтение пакетов... обработано: {evt.get('processed_packets')}")
            self.progress.setRange(0, 0)
        elif stage == 'parsing_done':
            total = evt.get('total_flows', 0)
            self.status_label.setText(f"Сформировано потоков: {total}")
        elif stage == 'scaling':
            total = evt.get('total_flows', 0)
            self.status_label.setText('Масштабирование признаков...')
            self.progress.setRange(0, 0)
        elif stage == 'predict':
            total = evt.get('total_flows', 0)
            self.status_label.setText(f"Предсказание... потоков: {total}")
            self.progress.setRange(0, 0)
        elif stage == 'done':
            self.status_label.setText(f"Готово. Найдено атак: {evt.get('attacks')} из {evt.get('total_flows')}")

    def on_result(self, res: dict):
        self.last_result = res
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self.export_attacks_btn.setEnabled(True)
        # fill table
        attacks = res.get('attacks', [])
        self.table.setRowCount(len(attacks))
        for i, rec in enumerate(attacks):
            idx_item = QtWidgets.QTableWidgetItem(str(rec['index']))
            idx_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(i, 0, idx_item)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(rec['key'])))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(rec['traffic_type'])))

    def on_error(self, text: str):
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, 'Ошибка', text)

    def export_all(self):
        if not self.last_result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить все потоки', 'all_flows.txt', 'Text Files (*.txt)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            for rec in self.last_result.get('all_flows', []):
                f.write(format_flow_txt(rec) + '\n')
        QtWidgets.QMessageBox.information(self, 'Готово', f'Сохранено: {path}')

    def export_attacks(self):
        if not self.last_result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить атаки', 'attacks.txt', 'Text Files (*.txt)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            for rec in self.last_result.get('attacks', []):
                f.write(format_flow_txt(rec) + '\n')
        QtWidgets.QMessageBox.information(self, 'Готово', f'Сохранено: {path}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


