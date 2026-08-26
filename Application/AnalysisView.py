import sys
import numpy as np
from scipy.io import wavfile
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QToolBar,
    QHBoxLayout,
)
from PyQt6.QtGui import QAction
import pyqtgraph as pg


def load_audio(filepath):
    """Load a WAV file and return normalized channel data."""
    sample_rate, data = wavfile.read(filepath)

    # If mono, duplicate to two channels
    if data.ndim == 1:
        data = np.column_stack([data, data])

    # Normalize to float in range -1.0 to 1.0
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
    else:
        max_val = 1.0
    data = data.astype(np.float64) / max_val

    time_axis = np.arange(data.shape[0]) / sample_rate
    return sample_rate, time_axis, data[:, 0], data[:, 1]


def compute_spectrum(signal, sample_rate, max_freq=20000):
    """Compute the frequency spectrum of a signal."""
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    fft_data = np.fft.rfft(signal)
    magnitude = np.abs(fft_data) / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    mask = frequencies <= max_freq
    return frequencies[mask], magnitude[mask]


class AudioEditorWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sample_rate = None
        self.time_axis = None
        self.left_data = None
        self.right_data = None

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout(self)

        load_button = QPushButton("Choose File")
        load_button.clicked.connect(self.open_file)
        layout.addWidget(load_button, stretch=0)

        sub_layout = QHBoxLayout()
        layout.addLayout(sub_layout)

        # Use a dark background for a professional audio editor look
        pg.setConfigOption("background", "k")
        pg.setConfigOption("foreground", "w")

        # Spectrum plot (top)
        self.spectrum_plot = pg.PlotWidget(title="Frequency Spectrum")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Magnitude")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.spectrum_plot)

        # Left channel waveform
        self.left_plot = pg.PlotWidget(title="Left Channel")
        self.left_plot.setLabel("bottom", "Time", units="s")
        self.left_plot.setLabel("left", "Amplitude")
        self.left_plot.showGrid(x=True, y=True, alpha=0.3)
        sub_layout.addWidget(self.left_plot)

        # Right channel waveform
        self.right_plot = pg.PlotWidget(title="Right Channel")
        self.right_plot.setLabel("bottom", "Time", units="s")
        self.right_plot.setLabel("left", "Amplitude")
        self.right_plot.showGrid(x=True, y=True, alpha=0.3)
        sub_layout.addWidget(self.right_plot)

        # Link the X axes of left and right channels
        self.right_plot.setXLink(self.left_plot)
        self.right_plot.setYLink(self.left_plot)

        # Region selector on the left channel plot
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)  # Draw on top of the waveform
        self.left_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self.update_spectrum)

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open WAV File", "", "WAV Files (*.wav)"
        )
        if filepath:
            self.load_and_display(filepath)

    def load_and_display(self, filepath):
        self.sample_rate, self.time_axis, self.left_data, self.right_data = load_audio(
            filepath
        )

        # Downsample for display if the file is very large.
        # PyQtGraph handles large datasets well, but downsampling
        # speeds up the initial render for very long files.
        display_left = self.left_data
        display_right = self.right_data
        display_time = self.time_axis

        max_display_points = 500_000
        if len(self.time_axis) > max_display_points:
            step = len(self.time_axis) // max_display_points
            display_time = self.time_axis[::step]
            display_left = self.left_data[::step]
            display_right = self.right_data[::step]

        # Plot waveforms
        self.left_plot.clear()
        self.left_plot.plot(display_time, display_left, pen=pg.mkPen("c", width=1))
        self.left_plot.addItem(self.region)

        self.right_plot.clear()
        self.right_plot.plot(display_time, display_right, pen=pg.mkPen("m", width=1))

        # Set region to the first 10% of the file (or the whole thing if short)
        duration = self.time_axis[-1]
        region_end = min(duration, duration * 0.1)
        self.region.setRegion([0, region_end])

        # Initial spectrum update
        self.update_spectrum()

    def update_spectrum(self):
        if self.left_data is None:
            return

        # Get the selected region bounds
        min_time, max_time = self.region.getRegion()

        # Convert time bounds to sample indices
        start_idx = max(0, int(min_time * self.sample_rate))
        end_idx = min(len(self.left_data), int(max_time * self.sample_rate))

        if end_idx <= start_idx:
            return

        # Use the left channel for the spectrum (you could average both)
        segment = self.left_data[start_idx:end_idx]
        frequencies, magnitude = compute_spectrum(segment, self.sample_rate)

        # Convert magnitude to dB for a more useful display
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        self.spectrum_plot.clear()
        self.spectrum_plot.plot(frequencies, magnitude_db, pen=pg.mkPen("y", width=1))


def main():
    app = QApplication(sys.argv)
    window = AudioEditorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
