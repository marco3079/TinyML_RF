from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import os
import re
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QProcess, QTimer, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QPalette, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "model" / "dataset" / "rostos"
MODEL_HEADER = "src/model_data.h"
MODEL_CHECKPOINT = "src/model_data.keras"
SERIAL_PORT_TITLE = "Porta serial"
DEFAULT_SERIAL_PORT = "COM8"
UPLOAD_TASK = "Gravando placa"


def default_training_python() -> str:
    candidates = (
        PROJECT_ROOT / ".training-env" / "python.exe",
        PROJECT_ROOT / ".training-env" / "Scripts" / "python.exe",
    )
    return str(next((path for path in candidates if path.exists()), Path(sys.executable)))


def find_platformio() -> str:
    for name in ("pio", "platformio"):
        executable = shutil.which(name)
        if executable:
            return executable
    candidates = [
        Path.home() / ".platformio" / "penv" / "Scripts" / "platformio.exe",
    ]
    app_data = Path(os.environ.get("APPDATA", ""))
    if app_data.exists():
        candidates.extend(app_data.glob("Python/Python*/Scripts/pio.exe"))
    return str(next((path for path in candidates if path.exists()), ""))


def count_dataset(path: Path) -> tuple[int, int]:
    images = sum(1 for suffix in ("*.jpg", "*.jpeg", "*.png") for _ in path.rglob(suffix))
    labels = sum(1 for _ in path.rglob("*.txt"))
    return images, labels


class CommandRunner:
    def __init__(self, output: Callable[[str], None], finished: Callable[[bool], None]):
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.output = output
        self.finished = finished

    def start(self, executable: str, arguments: list[str]) -> None:
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.start(executable, arguments)

    def cancel(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _read(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.output(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read()
        self.finished(exit_code == 0)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TinyML Studio - XIAO ESP32S3")
        self.resize(1180, 860)
        self.setMinimumSize(980, 800)
        self.runner = CommandRunner(self.append_log, self.task_finished)
        self.pipeline: deque[tuple[str, str, list[str]]] = deque()
        self.current_task = ""
        self.serial_connection = None
        self.serial_receive_buffer = ""
        self.serial_bytes_received = 0
        self.serial_bytes_sent = 0
        self.reconnect_serial_after_task = False
        self.serial_timer = QTimer(self)
        self.serial_timer.setInterval(30)
        self.serial_timer.timeout.connect(self.read_serial)
        self._build_ui()
        self._apply_style()
        self.refresh_dataset_summary()
        self.refresh_ports()
        self.refresh_model_state()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame(objectName="header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 20, 28, 20)
        title_box = QVBoxLayout()
        title = QLabel("TinyML Studio", objectName="title")
        subtitle = QLabel("Treino e deploy para XIAO ESP32S3 Sense", objectName="subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        self.model_badge = QLabel("MODELO NAO TREINADO", objectName="badge")
        header_layout.addWidget(self.model_badge)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        rail = QFrame(objectName="rail")
        rail.setFixedWidth(230)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(22, 28, 22, 24)
        rail_layout.setSpacing(10)
        rail_layout.addWidget(QLabel("PIPELINE", objectName="eyebrow"))
        self.step_labels = []
        for number, text in enumerate(
            ("Validar dataset", "Treinar modelo", "Compilar modelo", "Firmware e upload"), 1
        ):
            label = QLabel(f"{number:02d}  {text}", objectName="step")
            self.step_labels.append(label)
            rail_layout.addWidget(label)
        rail_layout.addStretch()
        docs = QPushButton("Abrir documentacao", objectName="quietButton")
        docs.setIcon(self.style().standardIcon(QStyle.SP_DialogHelpButton))
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(PROJECT_ROOT / "README.md"))))
        rail_layout.addWidget(docs)
        body.addWidget(rail)

        workspace = QWidget(objectName="workspace")
        content = QVBoxLayout(workspace)
        content.setContentsMargins(28, 24, 28, 24)
        content.setSpacing(16)
        body.addWidget(workspace, 1)

        dataset_panel = QFrame(objectName="panel")
        dataset_layout = QGridLayout(dataset_panel)
        dataset_layout.setContentsMargins(20, 18, 20, 18)
        dataset_layout.setHorizontalSpacing(12)
        dataset_layout.addWidget(QLabel("Dataset", objectName="panelTitle"), 0, 0, 1, 3)
        self.dataset_path = QLineEdit(str(DEFAULT_DATASET))
        browse_dataset = QPushButton("Selecionar")
        browse_dataset.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_dataset.clicked.connect(self.choose_dataset)
        self.dataset_summary = QLabel(objectName="meta")
        dataset_layout.addWidget(self.dataset_path, 1, 0, 1, 2)
        dataset_layout.addWidget(browse_dataset, 1, 2)
        dataset_layout.addWidget(self.dataset_summary, 2, 0, 1, 3)
        content.addWidget(dataset_panel)

        settings_panel = QFrame(objectName="panel")
        settings_layout = QGridLayout(settings_panel)
        settings_layout.setContentsMargins(20, 18, 20, 18)
        settings_layout.setHorizontalSpacing(12)
        settings_layout.setVerticalSpacing(10)
        settings_layout.addWidget(QLabel("Configuracao", objectName="panelTitle"), 0, 0, 1, 6)
        settings_layout.addWidget(QLabel("Python de treino", objectName="fieldLabel"), 1, 0, 1, 3)
        settings_layout.addWidget(QLabel(SERIAL_PORT_TITLE, objectName="fieldLabel"), 1, 3, 1, 3)
        self.python_path = QLineEdit(default_training_python())
        choose_python = QPushButton("...")
        choose_python.setFixedWidth(42)
        choose_python.clicked.connect(self.choose_python)
        self.port_combo = QComboBox()
        refresh = QPushButton("Atualizar")
        refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh.clicked.connect(self.refresh_ports)
        settings_layout.addWidget(self.python_path, 2, 0, 1, 2)
        settings_layout.addWidget(choose_python, 2, 2)
        settings_layout.addWidget(self.port_combo, 2, 3, 1, 2)
        settings_layout.addWidget(refresh, 2, 5)
        self.epochs = self._spin(1, 500, 60)
        self.batch_size = self._spin(1, 256, 32)
        self.seed = self._spin(0, 999999, 42)
        for column, (label, widget) in enumerate(
            (("Epocas", self.epochs), ("Lote", self.batch_size), ("Semente", self.seed))
        ):
            settings_layout.addWidget(QLabel(label, objectName="fieldLabel"), 3, column * 2, 1, 2)
            settings_layout.addWidget(widget, 4, column * 2, 1, 2)
        content.addWidget(settings_panel)

        actions = QHBoxLayout()
        self.validate_button = self._action("Validar", QStyle.SP_DialogApplyButton, self.validate_dataset)
        self.train_button = self._action("Treinar", QStyle.SP_MediaPlay, self.train_model)
        self.model_button = self._action("Compilar modelo", QStyle.SP_DriveFDIcon, self.compile_model)
        self.build_button = self._action("Compilar firmware", QStyle.SP_ComputerIcon, self.build_firmware)
        self.upload_button = self._action("Gravar placa", QStyle.SP_ArrowUp, self.upload_firmware)
        for button in (
            self.validate_button,
            self.train_button,
            self.model_button,
            self.build_button,
            self.upload_button,
        ):
            actions.addWidget(button)
        self.pipeline_button = QPushButton("Executar pipeline", objectName="primaryButton")
        self.pipeline_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.pipeline_button.clicked.connect(self.run_pipeline)
        actions.addWidget(self.pipeline_button)
        content.addLayout(actions)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Pronto", objectName="statusLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.cancel_button = QPushButton("Cancelar", objectName="dangerButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        status_row.addWidget(self.status_label)
        status_row.addWidget(self.progress, 1)
        status_row.addWidget(self.cancel_button)
        content.addLayout(status_row)

        self.tabs = QTabWidget(objectName="tabs")
        self.log = QTextEdit(objectName="log")
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Cascadia Mono", 9))
        self.tabs.addTab(self.log, "Log do pipeline")
        self.tabs.addTab(self._build_serial_panel(), "Monitor serial")
        content.addWidget(self.tabs, 1)

    def _build_serial_panel(self) -> QWidget:
        panel = QWidget(objectName="serialPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        connection_row = QHBoxLayout()
        self.serial_status = QLabel("DESCONECTADO", objectName="serialStatus")
        self.baud_combo = QComboBox()
        for baud in (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600):
            self.baud_combo.addItem(str(baud), baud)
        self.baud_combo.setCurrentText("115200")
        self.serial_connect_button = QPushButton("Conectar")
        self.serial_connect_button.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.serial_connect_button.clicked.connect(self.toggle_serial)
        reset_button = QPushButton("Reset")
        reset_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reset_button.clicked.connect(self.reset_board)
        clear_button = QPushButton("Limpar")
        clear_button.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        clear_button.clicked.connect(lambda: self.serial_output.clear())
        save_button = QPushButton("Salvar log")
        save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_button.clicked.connect(self.save_serial_log)
        connection_row.addWidget(self.serial_status)
        connection_row.addStretch()
        connection_row.addWidget(QLabel("Baud", objectName="fieldLabel"))
        connection_row.addWidget(self.baud_combo)
        connection_row.addWidget(self.serial_connect_button)
        connection_row.addWidget(reset_button)
        connection_row.addWidget(clear_button)
        connection_row.addWidget(save_button)
        layout.addLayout(connection_row)

        option_row = QHBoxLayout()
        self.serial_timestamps = QCheckBox("Timestamps")
        self.serial_timestamps.setChecked(True)
        self.serial_autoscroll = QCheckBox("Auto-scroll")
        self.serial_autoscroll.setChecked(True)
        self.serial_hex_receive = QCheckBox("Receber em HEX")
        self.serial_counters = QLabel("RX 0 B  |  TX 0 B", objectName="meta")
        option_row.addWidget(self.serial_timestamps)
        option_row.addWidget(self.serial_autoscroll)
        option_row.addWidget(self.serial_hex_receive)
        option_row.addStretch()
        option_row.addWidget(self.serial_counters)
        layout.addLayout(option_row)

        self.serial_output = QTextEdit(objectName="serialOutput")
        self.serial_output.setReadOnly(True)
        self.serial_output.setFont(QFont("Cascadia Mono", 9))
        layout.addWidget(self.serial_output, 1)

        send_row = QHBoxLayout()
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("Digite um comando para a placa")
        self.serial_input.returnPressed.connect(self.send_serial_command)
        self.line_ending = QComboBox()
        self.line_ending.addItem("Nova linha (LF)", "\n")
        self.line_ending.addItem("CR + LF", "\r\n")
        self.line_ending.addItem("Nenhum", "")
        self.serial_hex_send = QCheckBox("Enviar HEX")
        ip_button = QPushButton("Pedir IP")
        ip_button.clicked.connect(lambda: self.send_serial_text("ip"))
        send_button = QPushButton("Enviar", objectName="serialSendButton")
        send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        send_button.clicked.connect(self.send_serial_command)
        send_row.addWidget(self.serial_input, 1)
        send_row.addWidget(self.line_ending)
        send_row.addWidget(self.serial_hex_send)
        send_row.addWidget(ip_button)
        send_row.addWidget(send_button)
        layout.addLayout(send_row)
        return panel

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _action(self, text: str, icon: QStyle.StandardPixmap, callback) -> QPushButton:
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(callback)
        return button

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            * { font-family: "Segoe UI"; font-size: 13px; color: #18211d; }
            QMainWindow, #workspace { background: #f2f1ec; }
            #header { background: #163b31; border-bottom: 4px solid #df5b3f; }
            #title { color: #ffffff; font: 600 27px "Segoe UI"; }
            #subtitle { color: #b9d0c8; font-size: 12px; }
            #badge { color: #ffffff; background: #a94431; padding: 7px 11px; border-radius: 3px; font-weight: 600; }
            #rail { background: #e4e2da; border-right: 1px solid #c8c5ba; }
            #eyebrow { color: #707970; font-size: 11px; font-weight: 700; }
            #step { padding: 10px 4px; border-bottom: 1px solid #cbc8be; font-weight: 600; }
            #panel { background: #fbfaf6; border: 1px solid #cac7bc; border-radius: 6px; }
            #panelTitle { font-size: 17px; font-weight: 650; padding-bottom: 5px; }
            #fieldLabel, #meta { color: #667069; font-size: 11px; }
            QLineEdit, QComboBox, QSpinBox { background: #ffffff; border: 1px solid #aaa99f; border-radius: 4px; padding: 7px; min-height: 20px; }
            QPushButton { background: #e8e6df; border: 1px solid #aaa99f; border-radius: 4px; padding: 8px 11px; font-weight: 600; }
            QPushButton:hover { background: #dcd9cf; border-color: #73766f; }
            QPushButton:disabled { color: #999b96; background: #ecebe7; }
            #primaryButton { background: #df5b3f; color: white; border-color: #bf442d; }
            #primaryButton:hover { background: #c94b32; }
            #dangerButton { color: #a02f25; }
            #quietButton { text-align: left; background: transparent; border: 0; padding: 8px 0; }
            #statusLabel { font-weight: 650; min-width: 125px; }
            QProgressBar { border: 1px solid #aaa99f; border-radius: 3px; background: #ffffff; height: 18px; text-align: center; }
            QProgressBar::chunk { background: #28785e; }
            #log { background: #171c1a; color: #cfe2d9; border: 0; border-radius: 5px; padding: 10px; selection-background-color: #3b6657; }
            #tabs::pane { border: 1px solid #aaa99f; border-radius: 4px; background: #fbfaf6; }
            QTabBar::tab { background: #dedbd2; border: 1px solid #aaa99f; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background: #fbfaf6; border-bottom-color: #fbfaf6; color: #163b31; }
            #serialPanel { background: #fbfaf6; }
            #serialOutput { background: #101513; color: #bfe8d7; border: 0; border-radius: 4px; padding: 9px; selection-background-color: #3b6657; }
            #serialStatus { color: #ffffff; background: #8c3c31; border-radius: 3px; padding: 6px 9px; font-weight: 700; }
            #serialSendButton { background: #28785e; color: white; border-color: #1e654d; }
        """)

    def choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Selecionar dataset", self.dataset_path.text())
        if path:
            self.dataset_path.setText(path)
            self.refresh_dataset_summary()

    def choose_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Python", str(Path(sys.executable).parent), "Python (python.exe)")
        if path:
            self.python_path.setText(path)

    def refresh_dataset_summary(self) -> None:
        path = Path(self.dataset_path.text())
        if not path.exists():
            self.dataset_summary.setText("Pasta nao encontrada")
            return
        images, labels = count_dataset(path)
        self.dataset_summary.setText(f"{images} imagens  |  {labels} rotulos YOLO")

    def refresh_model_state(self) -> None:
        header = PROJECT_ROOT / "src" / "model_data.h"
        trained = header.exists() and "constexpr bool trained = true" in header.read_text(errors="ignore")
        self.model_badge.setText("MODELO PRONTO" if trained else "MODELO NAO TREINADO")
        self.model_badge.setStyleSheet("background:#28785e;" if trained else "")

    def refresh_ports(self) -> None:
        selected = (
            self.serial_connection.port
            if self.serial_connection
            else self.port_combo.currentData() or DEFAULT_SERIAL_PORT
        )
        self.port_combo.clear()
        try:
            from serial.tools import list_ports

            ports = list(list_ports.comports())
        except ImportError:
            ports = []
        for port in ports:
            self.port_combo.addItem(f"{port.device}  {port.description}", port.device)
        if not ports:
            self.port_combo.addItem("Nenhuma placa detectada", "")
        index = self.port_combo.findData(selected)
        if index >= 0:
            self.port_combo.setCurrentIndex(index)

    def toggle_serial(self) -> None:
        if self.serial_connection:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self) -> None:
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, SERIAL_PORT_TITLE, "Selecione uma porta serial valida.")
            return
        try:
            import serial

            self.serial_connection = serial.Serial(
                port=port,
                baudrate=int(self.baud_combo.currentData()),
                timeout=0,
                write_timeout=1,
            )
        except Exception as error:
            self.serial_connection = None
            QMessageBox.critical(
                self,
                "Falha ao conectar",
                f"Nao foi possivel abrir {port}.\n\n"
                "Feche outros monitores seriais e tente novamente.\n\n"
                f"Detalhe: {error}",
            )
            return
        self.serial_receive_buffer = ""
        self.serial_timer.start()
        self.serial_connect_button.setText("Desconectar")
        self.serial_status.setText(f"CONECTADO {port}")
        self.serial_status.setStyleSheet("background:#28785e;")
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self.append_serial_message(f"Conectado a {port} em {self.serial_connection.baudrate} baud", "SYS")

    def disconnect_serial(self) -> None:
        self.serial_timer.stop()
        connection = self.serial_connection
        self.serial_connection = None
        if connection:
            try:
                connection.close()
            except Exception:
                pass
            self.append_serial_message("Porta serial desconectada", "SYS")
        self.serial_connect_button.setText("Conectar")
        self.serial_status.setText("DESCONECTADO")
        self.serial_status.setStyleSheet("")
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)

    def read_serial(self) -> None:
        if not self.serial_connection:
            return
        try:
            available = self.serial_connection.in_waiting
            if available <= 0:
                return
            data = self.serial_connection.read(available)
        except Exception as error:
            self.append_serial_message(f"Erro de leitura: {error}", "ERRO")
            self.disconnect_serial()
            return
        self.serial_bytes_received += len(data)
        self.update_serial_counters()
        if self.serial_hex_receive.isChecked():
            self.append_serial_message(data.hex(" ").upper(), "RX")
            return

        text = data.decode("utf-8", errors="replace").replace("\r", "")
        if not self.serial_timestamps.isChecked():
            self.serial_output.moveCursor(QTextCursor.End)
            self.serial_output.insertPlainText(text)
            self.scroll_serial_output()
            return
        self.serial_receive_buffer += text
        while "\n" in self.serial_receive_buffer:
            line, self.serial_receive_buffer = self.serial_receive_buffer.split("\n", 1)
            self.append_serial_message(line, "RX")

    def send_serial_command(self) -> None:
        self.send_serial_text(self.serial_input.text())

    def send_serial_text(self, text: str) -> None:
        if not self.serial_connection:
            QMessageBox.warning(self, "Monitor serial", "Conecte a porta antes de enviar comandos.")
            return
        if self.serial_hex_send.isChecked():
            try:
                payload = bytes.fromhex(text.replace("0x", "").replace(",", " "))
            except ValueError:
                QMessageBox.warning(self, "HEX invalido", "Use pares hexadecimais, por exemplo: 69 70")
                return
        else:
            payload = text.encode("utf-8")
        payload += self.line_ending.currentData().encode("ascii")
        try:
            written = self.serial_connection.write(payload)
            self.serial_connection.flush()
        except Exception as error:
            QMessageBox.critical(self, "Falha ao enviar", str(error))
            return
        self.serial_bytes_sent += written
        self.update_serial_counters()
        display = payload.hex(" ").upper() if self.serial_hex_send.isChecked() else text
        self.append_serial_message(display, "TX")
        self.serial_input.clear()

    def append_serial_message(self, text: str, direction: str) -> None:
        prefix = f"[{direction}]"
        if self.serial_timestamps.isChecked():
            prefix = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {prefix}"
        self.serial_output.append(f"{prefix} {text}")
        self.scroll_serial_output()

    def scroll_serial_output(self) -> None:
        if self.serial_autoscroll.isChecked():
            self.serial_output.moveCursor(QTextCursor.End)

    def update_serial_counters(self) -> None:
        self.serial_counters.setText(
            f"RX {self.serial_bytes_received} B  |  TX {self.serial_bytes_sent} B"
        )

    def reset_board(self) -> None:
        if not self.serial_connection:
            QMessageBox.warning(self, "Reset", "Conecte a porta serial primeiro.")
            return
        try:
            self.serial_connection.dtr = False
            self.serial_connection.rts = True
            QTimer.singleShot(100, self.finish_board_reset)
            self.append_serial_message("Pulso de reset enviado", "SYS")
        except Exception as error:
            QMessageBox.critical(self, "Falha no reset", str(error))

    def finish_board_reset(self) -> None:
        if not self.serial_connection:
            return
        try:
            self.serial_connection.rts = False
            self.serial_connection.dtr = True
        except Exception as error:
            self.append_serial_message(f"Falha ao concluir reset: {error}", "ERRO")

    def save_serial_log(self) -> None:
        default_name = PROJECT_ROOT / ".log" / f"serial-{datetime.now():%Y%m%d-%H%M%S}.txt"
        default_name.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar log serial", str(default_name), "Texto (*.txt);;Todos os arquivos (*)"
        )
        if path:
            Path(path).write_text(self.serial_output.toPlainText() + "\n", encoding="utf-8")

    def prepare_serial_for_upload(self) -> None:
        self.reconnect_serial_after_task = self.serial_connection is not None
        if self.serial_connection:
            self.disconnect_serial()

    def reconnect_serial_after_upload(self) -> None:
        if not self.reconnect_serial_after_task:
            return
        self.reconnect_serial_after_task = False
        self.refresh_ports()
        self.connect_serial()

    def python_command(self) -> str:
        return self.python_path.text().strip()

    def pio_command(self) -> str:
        command = find_platformio()
        if not command:
            raise FileNotFoundError("PlatformIO nao encontrado. Instale PlatformIO Core ou a extensao do VS Code.")
        return command

    def validate_dataset(self) -> None:
        self.start_task("Validando dataset", self.python_command(), [
            "model/train.py", "--dataset", self.dataset_path.text(), "--validate-only"
        ])

    def train_model(self) -> None:
        self.start_task("Treinando modelo", self.python_command(), self.training_arguments())

    def training_arguments(self) -> list[str]:
        return [
            "model/train.py", "--dataset", self.dataset_path.text(),
            "--output", MODEL_HEADER, "--epochs", str(self.epochs.value()),
            "--batch-size", str(self.batch_size.value()), "--seed", str(self.seed.value()),
        ]

    def compile_model(self) -> None:
        self.start_task("Compilando modelo", self.python_command(), [
            "model/compile_model.py", "--model", MODEL_CHECKPOINT, "--output", MODEL_HEADER
        ])

    def build_firmware(self) -> None:
        try:
            self.start_task("Compilando firmware", self.pio_command(), ["run"])
        except FileNotFoundError as error:
            QMessageBox.critical(self, "PlatformIO", str(error))

    def upload_firmware(self) -> None:
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, SERIAL_PORT_TITLE, "Conecte a placa e clique em Atualizar.")
            return
        try:
            pio = self.pio_command()
            self.prepare_serial_for_upload()
            self.start_task(UPLOAD_TASK, pio, ["run", "--target", "upload", "--upload-port", port])
        except FileNotFoundError as error:
            QMessageBox.critical(self, "PlatformIO", str(error))

    def run_pipeline(self) -> None:
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, SERIAL_PORT_TITLE, "Selecione a porta da placa antes de executar o pipeline.")
            return
        try:
            pio = self.pio_command()
        except FileNotFoundError as error:
            QMessageBox.critical(self, "PlatformIO", str(error))
            return
        python = self.python_command()
        self.pipeline = deque([
            ("Treinando modelo", python, self.training_arguments()),
            ("Compilando modelo", python, ["model/compile_model.py", "--model", MODEL_CHECKPOINT, "--output", MODEL_HEADER]),
            ("Compilando firmware", pio, ["run"]),
            (UPLOAD_TASK, pio, ["run", "--target", "upload", "--upload-port", port]),
        ])
        self.start_next_pipeline_task()

    def start_next_pipeline_task(self) -> None:
        if not self.pipeline:
            self.status_label.setText("Pipeline concluido")
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.set_busy(False)
            return
        name, executable, arguments = self.pipeline.popleft()
        if name == UPLOAD_TASK:
            self.prepare_serial_for_upload()
        self.start_task(name, executable, arguments)

    def start_task(self, name: str, executable: str, arguments: list[str]) -> None:
        if self.runner.process.state() != QProcess.NotRunning:
            return
        if not executable or not Path(executable).exists():
            QMessageBox.critical(self, "Executavel", f"Executavel nao encontrado:\n{executable}")
            self.pipeline.clear()
            return
        self.current_task = name
        self.status_label.setText(name)
        self.progress.setRange(0, 0)
        self.append_log(f"\n$ {executable} {' '.join(arguments)}")
        self.set_busy(True)
        self.runner.start(executable, arguments)

    def append_log(self, line: str) -> None:
        self.log.append(line.replace("\r", ""))
        match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
        if match:
            current, total = map(int, match.groups())
            self.progress.setRange(0, total)
            self.progress.setValue(current)

    def task_finished(self, success: bool) -> None:
        self.append_log("[OK]" if success else "[FALHA]")
        self.refresh_model_state()
        upload_finished = self.current_task == UPLOAD_TASK
        if not success:
            self.pipeline.clear()
            self.status_label.setText("Falha")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.set_busy(False)
            if upload_finished:
                QTimer.singleShot(1200, self.reconnect_serial_after_upload)
            return
        if self.pipeline:
            self.start_next_pipeline_task()
        else:
            self.status_label.setText("Concluido")
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.set_busy(False)
        if upload_finished:
            QTimer.singleShot(1200, self.reconnect_serial_after_upload)

    def cancel_task(self) -> None:
        self.pipeline.clear()
        self.runner.cancel()
        self.append_log("[CANCELADO]")

    def set_busy(self, busy: bool) -> None:
        for button in (
            self.validate_button, self.train_button, self.model_button,
            self.build_button, self.upload_button, self.pipeline_button,
        ):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def closeEvent(self, event) -> None:
        self.runner.cancel()
        self.disconnect_serial()
        event.accept()


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if windows_font.exists():
        QFontDatabase.addApplicationFont(str(windows_font))
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    palette.setColor(QPalette.Highlight, QColor("#28785e"))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
