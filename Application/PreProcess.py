import os
import glob
import random
 
import torch
import torchaudio
import torchaudio.transforms as transforms
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def load_audio(file_path):
    
    data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); torchaudio convention is (channels, samples)
    waveform = torch.from_numpy(data.T)
    return waveform, sample_rate


def preprocess_audio(waveform, sample_rate, target_sample_rate=16000):
    """
    Preprocess the audio waveform by resampling and normalizing.

    Args:
        waveform (Tensor): The audio waveform.
        sample_rate (int): The original sample rate of the audio.
        target_sample_rate (int): The desired sample rate for the output waveform.
    """
    # Resample if the sample rate is different from the target
    if sample_rate != target_sample_rate:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono through averages
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Normalize to [-1, 1]
    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        waveform = waveform / peak


    return waveform, target_sample_rate

def compute_stft(waveform, n_fft=512, hop_length=128, win_length=None):
    win_length = win_length if win_length is not None else n_fft
    window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform, 
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    magnitude = magnitude[..., :-1, :]
    phase = phase[..., :-1, :]
    return magnitude, phase

def magnitude_to_log(magnitude, eps=1e-5):
    return torch.log(magnitude + eps)

def log_to_magnitude(log_magnitude, eps=1e-5):
    return torch.exp(log_magnitude) - eps

def reconstruct_waveform(magnitude, phase, n_fft=512, hop_length=128, win_length=None):
    # Remove extra channel dimension if present (e.g. [B, 1, F, T] -> [B, F, T])
    if magnitude.dim() == 4 and magnitude.size(1) == 1:
        magnitude = magnitude.squeeze(1)
    if phase.dim() == 4 and phase.size(1) == 1:
        phase = phase.squeeze(1)

    magnitude = torch.nn.functional.pad(magnitude, (0, 0, 0, 1))  # pad freq axis by 1 zero row
    phase = torch.nn.functional.pad(phase, (0, 0, 0, 1))

    win_length = win_length if win_length is not None else n_fft
    window = torch.hann_window(win_length)
    complex_stft = torch.polar(magnitude, phase)
    waveform = torch.istft(
        complex_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    return waveform

def pair_noisy_clean_files(noisy_dir, clean_dir):
    """
    Match noisy and clean audio files based on their filenames.
    
    Noisy dataset from UoE stores files under the same filename
    """
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    pairs = []
    for noisy_path in noisy_files: 
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, filename)
        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
        else:
            print(f"Warning: Clean file not found for {filename}")

    return pairs


def split_pairs_by_speaker(pairs, val_speakers=("p226", "p287")):
   
    train_pairs, val_pairs = [], []
    for noisy_path, clean_path in pairs:
        speaker = os.path.basename(noisy_path).split("_")[0]
        if speaker in val_speakers:
            val_pairs.append((noisy_path, clean_path))
        else:
            train_pairs.append((noisy_path, clean_path))
    return train_pairs, val_pairs


class DenoisingWaveformDataset(Dataset):
    """
    Dataset for waveform based Denoising

    e.g. DEMUCS / RTSEWD

    Each item is cropped/padded to a fixed number of samples (segment_samples)
    so that variable-length utterances can be batched together. This mirrors
    the segment_frames logic used in DenoisingSTFTDataset, but operates in the
    raw waveform domain instead of on STFT frames.

    segment_samples must be divisible by 2**(number of downsampling stages)
    in the waveform model (e.g. 16 for the default 4-stage RTSEWD), so that
    the encoder/decoder path lines up exactly without needing to interpolate.
    """

    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, segment_samples=16384, pairs=None):
        self.pairs = pairs if pairs is not None else pair_noisy_clean_files(noisy_dir, clean_dir)
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples

    def __len__(self):
        return len(self.pairs)

    def _load_and_preprocess(self, path):
        waveform, sr = load_audio(path)
        waveform, sr = preprocess_audio(waveform, sr, self.sample_rate)
        return waveform

    def __getitem__(self, index):
        noisy_path, clean_path = self.pairs[index]
        noisy_waveform = self._load_and_preprocess(noisy_path)
        clean_waveform = self._load_and_preprocess(clean_path)

        
        min_len = min(noisy_waveform.shape[-1], clean_waveform.shape[-1])
        noisy_waveform = noisy_waveform[..., :min_len]
        clean_waveform = clean_waveform[..., :min_len]

        total_len = noisy_waveform.shape[-1]

        if total_len >= self.segment_samples:
            
            start = random.randint(0, total_len - self.segment_samples)
            end = start + self.segment_samples
            noisy_waveform = noisy_waveform[..., start:end]
            clean_waveform = clean_waveform[..., start:end]
        else:
           
            pad_amount = self.segment_samples - total_len
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_amount))
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, pad_amount))

        return noisy_waveform, clean_waveform


class DenoisingSTFTDataset(Dataset):
    """
    Dataset for U-Net style denoising 
    
    Each item fixed size (in time frames)
    """

    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, n_fft=512, hop_length=128,
                 segment_frames=128, log_compress=True, pairs=None):
     
        self.pairs = pairs if pairs is not None else pair_noisy_clean_files(noisy_dir, clean_dir)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_frames = segment_frames
        self.log_compress = log_compress


    def __len__(self):
        return len(self.pairs)

    def _load_and_stft(self, path):
        waveform, sr = load_audio(path)
        waveform, sr = preprocess_audio(waveform, sr, self.sample_rate)
        # Ensure self.n_fft and self.hop_length are passed cleanly:
        magnitude, phase = compute_stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        return magnitude, phase

    def __getitem__(self, index):
        noisy_path, clean_path = self.pairs[index]
        noisy_magnitude, noisy_phase = self._load_and_stft(noisy_path)
        clean_magnitude, clean_phase = self._load_and_stft(clean_path)

        # Align time frames between clean and noisy audio
        min_frames = min(noisy_magnitude.shape[-1], clean_magnitude.shape[-1])
        noisy_magnitude = noisy_magnitude[..., :min_frames]
        clean_magnitude = clean_magnitude[..., :min_frames]
        noisy_phase = noisy_phase[..., :min_frames]

        total_frames = noisy_magnitude.shape[-1]

        if total_frames >= self.segment_frames:
            # Random crop along the time dimension
            start = random.randint(0, total_frames - self.segment_frames)
            end = start + self.segment_frames
            noisy_mag = noisy_magnitude[..., start:end]
            clean_mag = clean_magnitude[..., start:end]
            noisy_phase = noisy_phase[..., start:end]
        else:
            # Zero-pad the time dimension if shorter than segment_frames
            pad_amount = self.segment_frames - total_frames 
            # (pad_left, pad_right) applies to the LAST dimension (time)
            noisy_mag = torch.nn.functional.pad(noisy_magnitude, (0, pad_amount))
            clean_mag = torch.nn.functional.pad(clean_magnitude, (0, pad_amount))
            noisy_phase = torch.nn.functional.pad(noisy_phase, (0, pad_amount))
 
        
        noisy_mag_linear = noisy_mag
        clean_mag_linear = clean_mag

        if self.log_compress:
            noisy_mag_log = magnitude_to_log(noisy_mag)
        else:
            noisy_mag_log = noisy_mag

        return noisy_mag_log, noisy_mag_linear, clean_mag_linear, noisy_phase


def visualize_spectrogram(noisy_batch, clean_batch, title="Spectrogram"):
    sample_idx = 0
    noisy_spec = noisy_batch[sample_idx, 0].cpu().numpy()  # Shape: (257, 128)
    clean_spec = clean_batch[sample_idx, 0].cpu().numpy()  # Shape: (257, 128)
    noise_residual = noisy_spec - clean_spec               # Isolated noise component

    # 2. Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot Noisy Input
    im0 = axes[0].imshow(noisy_spec, origin='lower', aspect='auto', cmap='magma')
    axes[0].set_title('Noisy Log-Magnitude (Model Input)')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Frequency Bins')
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    # Plot Clean Target
    im1 = axes[1].imshow(clean_spec, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title('Clean Log-Magnitude (Target)')
    axes[1].set_xlabel('Time Frames')
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    # Plot Noise Difference
    im2 = axes[2].imshow(noise_residual, origin='lower', aspect='auto', cmap='coolwarm')
    axes[2].set_title('Noise Residual (Noisy - Clean)')
    axes[2].set_xlabel('Time Frames')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    noisy_dir = "Data/noisy_trainset_28spk_wav/noisy_trainset_28spk_wav"
    clean_dir = "Data/clean_trainset_28spk_wav/clean_trainset_28spk_wav"

    one_file = glob.glob(os.path.join(noisy_dir, "*.wav"))[0]
    waveform, sample_rate = load_audio(one_file)
    print(waveform.shape, sample_rate)  # Should be (1, N) for mono audio
 
    dataset = DenoisingSTFTDataset(noisy_dir, clean_dir)
    print(f"Found {len(dataset)} noisy/clean pairs")
 
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    noisy_log, noisy_linear, clean_linear, phase_batch = next(iter(loader))
 
    print("Noisy log-magnitude batch shape:", noisy_log.shape)  # (B, 1, freq_bins, segment_frames)
    print("Noisy linear-magnitude batch shape:", noisy_linear.shape)
    print("Clean linear-magnitude batch shape:", clean_linear.shape)
    print("Phase batch shape:", phase_batch.shape)

    visualize_spectrogram(noisy_log, magnitude_to_log(clean_linear))