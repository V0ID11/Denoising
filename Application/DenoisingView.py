import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QLineEdit,
)
import torch

from UNet import DenoisingUNet
from RTSEWD import DenoisingRTSEWD
from DenoiseController import *


class DenoiseSelectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Denoise")
        self.resize(1100, 750)

        self.setupui()

    def setupui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["UNet", "SEGAN"])

        layout.addWidget(self.model_selector)

        self.model_file = QLabel("")
        self.model_file_selector_button = QPushButton("Select Model File")
        self.model_file_selector_button.clicked.connect(self.model_file_selector)
        layout.addWidget(self.model_file)
        layout.addWidget(self.model_file_selector_button)

        self.audio_file = QLabel("")
        self.audio_file_selector_button = QPushButton("Select Audio File")
        self.audio_file_selector_button.clicked.connect(self.audio_file_selector)
        layout.addWidget(self.audio_file)
        layout.addWidget(self.audio_file_selector_button)

        self.output_path = QLineEdit("")
        self.output_path.setPlaceholderText("Select or enter output .wav file path")
        layout.addWidget(self.output_path)

        self.output_file_selector_button = QPushButton("Select Output File Location")
        self.output_file_selector_button.clicked.connect(self.output_file_selector)
        layout.addWidget(self.output_file_selector_button)
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirmed)
        layout.addWidget(self.confirm_button)

    def model_file_selector(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Model Files (*.pt)"
        )
        if filepath:
            self.model_file.setText(filepath)

    def audio_file_selector(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", "", "Wav Files (*.wav)"
        )
        if filepath:
            self.audio_file.setText(filepath)

    def output_file_selector(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Clean WAV File", "output.wav", "Wav Files (*.wav)"
        )
        if filepath:
            self.output_path.setText(filepath)

    def confirmed(self):
        model_path = self.model_file.text()
        audio_path = self.audio_file.text()
        model = self.model_selector.currentIndex()
        output_path = self.output_path.text()

        if model == 0:
            model = DenoisingUNet()
        else:
            model = DenoisingRTSEWD()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        denoise_and_save_wav(model, model_path, audio_path, output_path, device)


def main():
    app = QApplication(sys.argv)
    window = DenoiseSelectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
