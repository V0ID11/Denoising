import sys
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
)

import DenoisingView
import AnalysisView
import PesqView


class DashboardHomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupui()

    def setupui(self):

        layout = QVBoxLayout(self)

        self.analysisButton = QPushButton("Analysis View")
        layout.addWidget(self.analysisButton)

        self.denoiseButton = QPushButton("Denoising View")
        layout.addWidget(self.denoiseButton)

        self.pesqButton = QPushButton("Pesq View")
        layout.addWidget(self.pesqButton)


class DashboardView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Denoising Suite")
        self.resize(1100, 750)

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.home_widget = DashboardHomeWidget()
        self.stackedWidget.addWidget(self.home_widget)

        self.analysis_widget = AnalysisView.AudioEditorWindow()
        self.stackedWidget.addWidget(self.analysis_widget)

        self.denoise_widget = DenoisingView.DenoiseSelectorWindow()
        self.stackedWidget.addWidget(self.denoise_widget)

        self.pesq_widget = PesqView.PesqProcessorWidget()
        self.stackedWidget.addWidget(self.pesq_widget)

        self.home_widget.analysisButton.clicked.connect(self.goToAnalysisView)
        self.home_widget.denoiseButton.clicked.connect(self.goToDenoiseView)
        self.home_widget.pesqButton.clicked.connect(self.goToPesqView)

        self.add_navigation_bar()

    def add_navigation_bar(self):
        top_bar = QHBoxLayout()
        self.back_button = QPushButton("Back to Dashboard")
        self.back_button.clicked.connect(self.goToHomeView)
        self.back_button.hide()

        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch()

        wrapper = QWidget()
        main_layout = QVBoxLayout(wrapper)
        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.stackedWidget)
        self.setCentralWidget(wrapper)

    def goToHomeView(self):
        self.stackedWidget.setCurrentIndex(0)
        self.back_button.hide()

    def goToAnalysisView(self):
        self.stackedWidget.setCurrentWidget(self.analysis_widget)
        self.back_button.show()

    def goToDenoiseView(self):
        self.stackedWidget.setCurrentWidget(self.denoise_widget)
        self.back_button.show()

    def goToPesqView(self):
        self.stackedWidget.setCurrentWidget(self.pesq_widget)
        self.back_button.show()


def main():
    app = QApplication(sys.argv)
    window = DashboardView()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
