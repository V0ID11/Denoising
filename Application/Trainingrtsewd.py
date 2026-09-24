import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import soundfile as sf

from PreProcess import (
    load_audio,
    preprocess_audio,
    pair_noisy_clean_files,
    split_pairs_by_speaker,
    DenoisingWaveformDataset,
)
from RTSEWD import DenoisingRTSEWD


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for noisy_waveform, clean_waveform in loader:
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)

        optimizer.zero_grad()

        estimated_clean = model(noisy_waveform)
        loss = criterion(estimated_clean, clean_waveform)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy_waveform.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for noisy_waveform, clean_waveform in loader:
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)

        estimated_clean = model(noisy_waveform)
        loss = criterion(estimated_clean, clean_waveform)

        running_loss += loss.item() * noisy_waveform.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def save_denoised_wavs(model, pairs, device, sample_rate=16000,
                        output_dir="checkpoints_rtsewd/test_outputs"):
    """
    Run inference on each full utterance (not a fixed-size training crop)
    and save one .wav per input file.

    `pairs` is a list of (noisy_path, clean_path) tuples, e.g. val_pairs.
    Batch size is effectively 1 here since files vary in length -- each is
    padded individually to the nearest multiple of the model's total
    downsampling factor (2 ** number of encoder stages), run through the
    model, then cropped back to its real length before saving. Unlike the
    spectrogram model, there's no phase to re-attach here -- the model
    directly regresses the denoised waveform rather than predicting a mask.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating and saving .wav files to {output_dir}...")

    downsample_factor = 2 ** len(model.features)

    for noisy_path, _ in pairs:
        waveform, sr = load_audio(noisy_path)
        waveform, sr = preprocess_audio(waveform, sr, sample_rate)

        waveform = waveform.unsqueeze(0).to(device)  # (1, 1, T)

        total_len = waveform.shape[-1]
        pad_amount = (downsample_factor - total_len % downsample_factor) % downsample_factor
        if pad_amount > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        estimated_clean = model(waveform)
        estimated_clean = estimated_clean[..., :total_len]   # crop back to the real length

        filename = os.path.basename(noisy_path)
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, estimated_clean.squeeze(0).squeeze(0).cpu().numpy(), samplerate=sample_rate)

    print("Finished saving .wav files successfully!")


def main():
    # ---- config ----
    noisy_dir = "../noisy_trainset_28spk_wav"
    clean_dir = "../clean_trainset_28spk_wav"
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-2
    early_stop_patience = 10
    segment_samples = 16384          # ~1.024s at 16kHz; must be divisible by 2**len(features)
    checkpoint_path = "../checkpoints_rtsewd/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_pairs = pair_noisy_clean_files(noisy_dir, clean_dir)
    train_pairs, val_pairs = split_pairs_by_speaker(all_pairs)
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    train_dataset = DenoisingWaveformDataset(
        noisy_dir, clean_dir, segment_samples=segment_samples, pairs=train_pairs
    )
    val_dataset = DenoisingWaveformDataset(
        noisy_dir, clean_dir, segment_samples=segment_samples, pairs=val_pairs
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DenoisingRTSEWD().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  -> saved new best checkpoint (val loss {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"No improvement for {early_stop_patience} epochs -- stopping early at epoch {epoch}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_denoised_wavs(model, val_pairs, device, output_dir="../checkpoints_rtsewd/test_outputs")


if __name__ == "__main__":
    main()