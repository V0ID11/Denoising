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
    QListWidget,
)
import torch

from UNet import DenoisingUNet
from RTSEWD import DenoisingRTSEWD
from DenoiseController import *


class DenoiseSelectorWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupui()

    def setupui(self):

        layout = QVBoxLayout(self)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["UNet", "SEGAN"])

        layout.addWidget(self.model_selector)

        self.model_file = QLabel("")
        self.model_file_selector_button = QPushButton("Select Model File")
        self.model_file_selector_button.clicked.connect(self.model_file_selector)
        layout.addWidget(self.model_file)
        layout.addWidget(self.model_file_selector_button)

        self.audio_files_list = QListWidget()
        self.audio_file_selector_button = QPushButton("Select Audio File")
        self.audio_file_selector_button.clicked.connect(self.audio_file_selector)
        layout.addWidget(self.audio_files_list)
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
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Open WAV file", "", "Wav Files (*.wav)"
        )
        if filepaths:
            self.audio_files = filepaths
            self.audio_files_list.clear()
            self.audio_files_list.addItems([os.path.basename(p) for p in filepaths])

    def output_file_selector(self):
        dirpath = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if dirpath:
            self.output_path.setText(dirpath)

    def confirmed(self):
        model_path = self.model_file.text()

        if not self.audio_files:
            print("No audio files selected.")
            return

        model = self.model_selector.currentIndex()
        output_path = self.output_path.text()

        if not output_path or not os.path.exists(output_path):
            print("Invalid ouptut directory path.")
            return

        if model == 0:
            model = DenoisingUNet()
        else:
            model = DenoisingRTSEWD()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for in_file_path in self.audio_files:
            filename = os.path.basename(in_file_path)
            out_file_path = os.path.join(output_path, filename)

            denoise_and_save_wav(model, model_path, in_file_path, out_file_path, device)
        print("Batch processing complete")


def main():
    app = QApplication(sys.argv)
    window = DenoiseSelectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
