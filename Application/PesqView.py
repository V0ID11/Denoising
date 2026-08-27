import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QListView,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QCheckBox,
    QLabel,
    QListWidget,
    QLineEdit,
    QPushButton,
)

from DenoiseController import *


class PesqProcessorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize file tracking attributes
        self.clean_files = []
        self.noisy_files = []
        self.log_path = ""
        self.output_dict = {}

        # Set up a single main layout once
        self.main_layout = QVBoxLayout(self)

        self.setupui()

    def clear_layout(self):
        """Remove all existing widgets from the main layout."""
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def setupui(self):
        self.clear_layout()

        # Clean File Selector
        self.clean_file_selector_button = QPushButton("Select Clean Audio Files")
        self.clean_file_selector_button.clicked.connect(self.open_clean_files)
        self.main_layout.addWidget(self.clean_file_selector_button)

        # Noisy File Selector
        self.noisy_file_selector_button = QPushButton("Select Noisy Audio Files")
        self.noisy_file_selector_button.clicked.connect(self.open_noisy_files)
        self.main_layout.addWidget(self.noisy_file_selector_button)

        # Logging
        self.logging_check_box_selector = QCheckBox("Logging: ")
        self.logging_file_path_button = QPushButton("Select File for Logging output")
        self.logging_file_path_button.clicked.connect(self.get_output_path)
        self.logging_check_box_selector.clicked.connect(self.update_logging_selector)

        self.main_layout.addWidget(self.logging_check_box_selector)
        self.main_layout.addWidget(self.logging_file_path_button)

        self.logging_file_path_button.hide()

        # Continue
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.runPesq)
        self.main_layout.addWidget(self.continue_button)

    def displayPesqView(self):
        self.clear_layout()

        # Show All FileNames - Score
        self.pesq_list_view = QListWidget()

        for key, value in self.output_dict.items():
            self.pesq_list_view.addItem(f"{key} : {value}")

        self.main_layout.addWidget(self.pesq_list_view)

        # Return to setup
        self.return_button = QPushButton("Return to Selection View")
        self.return_button.clicked.connect(self.setupui)
        self.main_layout.addWidget(self.return_button)

    def update_logging_selector(self):
        if self.logging_check_box_selector.isChecked():
            self.logging_file_path_button.show()
        else:
            self.logging_file_path_button.hide()

    def open_clean_files(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Open WAV Files", "", "Wav Files (*.wav)"
        )
        if filepaths:
            self.clean_files = filepaths

    def open_noisy_files(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Open WAV Files", "", "Wav Files (*.wav)"
        )
        if filepaths:
            self.noisy_files = filepaths

    def get_output_path(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Select txt file", "", "TXT Files (*.txt)"
        )
        if filepath:
            self.log_path = filepath

    def runPesq(self):
        if not self.clean_files or not self.noisy_files:
            print("Please select both clean and noisy files.")
            return

        first_clean = os.path.basename(self.clean_files[0])
        first_noisy = os.path.basename(self.noisy_files[0])
        if len(self.clean_files) != len(self.noisy_files) or first_clean != first_noisy:
            print("Not same files")
            return

        self.output_dict = {}
        for clean_file, noisy_file in zip(self.clean_files, self.noisy_files):
            self.output_dict[os.path.basename(clean_file)] = calculate_pesq(
                clean_file, noisy_file
            )

        if self.logging_check_box_selector.isChecked() and self.log_path:
            with open(self.log_path, "w") as f:
                for key, value in self.output_dict.items():
                    f.write(f"{key}:{value}\n")

        self.displayPesqView()
