import os

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, RMSprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import soundfile as sf

import torchaudio

from PreProcess import *
from UNet import DenoisingUNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for noisy_log, noisy_linear, clean_linear, _ in loader:
        noisy_log = noisy_log.to(device)
        noisy_linear = noisy_linear.to(device)
        clean_linear = clean_linear.to(device)

        optimizer.zero_grad()

        mask = model(noisy_log)                      
        estimated_clean = mask * noisy_linear        
        loss = criterion(estimated_clean, clean_linear)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy_log.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for noisy_log, noisy_linear, clean_linear, _ in loader:
        noisy_log = noisy_log.to(device)
        noisy_linear = noisy_linear.to(device)
        clean_linear = clean_linear.to(device)

        mask = model(noisy_log)
        estimated_clean = mask * noisy_linear
        loss = criterion(estimated_clean, clean_linear)

        running_loss += loss.item() * noisy_log.size(0)

    return running_loss / len(loader.dataset)



@torch.no_grad()
def save_denoised_wavs(model, pairs, device, sample_rate=16000, n_fft=512,
                        hop_length=128, output_dir="checkpoints/test_outputs"):
    """
    Run inference on each full utterance (not a fixed-size training crop)
    and save one .wav per input file.

    `pairs` is a list of (noisy_path, clean_path) tuples, e.g. val_pairs.
    Batch size is effectively 1 here since files vary in length --
    each is padded individually to the nearest multiple of 16 (required
    by the 4 downsampling stages), run through the model, then cropped
    back to its real length before reconstruction.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating and saving .wav files to {output_dir}...")

    for noisy_path, _ in pairs:
        waveform, sr = load_audio(noisy_path)
        waveform, sr = preprocess_audio(waveform, sr, sample_rate)

        magnitude, phase = compute_stft(waveform, n_fft=n_fft, hop_length=hop_length)
        magnitude = magnitude.unsqueeze(0).to(device)  # (1, 1, 256, T)
        phase = phase.unsqueeze(0).to(device)

        total_frames = magnitude.shape[-1]
        pad_amount = (16 - total_frames % 16) % 16
        if pad_amount > 0:
            magnitude = torch.nn.functional.pad(magnitude, (0, pad_amount))

        noisy_log = magnitude_to_log(magnitude)
        mask = model(noisy_log)
        mask = mask[..., :total_frames]          # crop back to the real length
        magnitude = magnitude[..., :total_frames]

        estimated_clean = mask * magnitude
        reconstructed = reconstruct_waveform(
            estimated_clean.cpu(), phase.cpu(),
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        )

        filename = os.path.basename(noisy_path)
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, reconstructed.squeeze(0).numpy(), samplerate=sample_rate)

    print("Finished saving .wav files successfully!")


def train_epochs(model, 
                train_loader, 
                val_loader, 
                num_epochs,
                optimizer, 
                criterion, 
                scheduler, 
                early_stop_patience,
                device, 
                checkpoint_path, 
                best_val_loss=float("inf")):
    
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

    return best_val_loss
        

def hyperparameter_grid():
    learning_rates = [1e-4,5e-4,1e-3, 5e-3, 1e-2]
    weight_decays = [5e-3,1e-2, 5e-2,1e-1]
    optimizers = ["AdamW", "Adam", "RMSProp", "SGD"]
    batch_sizes = [8, 16, 32, 64]
    
    return learning_rates, weight_decays, optimizers, batch_sizes

def main():
    # ---- config ----
    noisy_dir = "../noisy_trainset_28spk_wav"
    clean_dir = "../clean_trainset_28spk_wav"
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-2         
    early_stop_patience = 10     
    checkpoint_path = "../checkpoints/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lrs, wds, optims, batchs = hyperparameter_grid()

    all_pairs = pair_noisy_clean_files(noisy_dir, clean_dir)
    train_pairs, val_pairs = split_pairs_by_speaker(all_pairs)
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    best_loss = float("inf")
    best_params = ()

    for lr in lrs:
        for wd in wds:
            for optim in optims:
                for batch in batchs:  
                    train_dataset = DenoisingSTFTDataset(noisy_dir, clean_dir, pairs=train_pairs)
                    val_dataset = DenoisingSTFTDataset(noisy_dir, clean_dir, pairs=val_pairs)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    
                    model = DenoisingUNet().to(device)

                    if optim == "AdamW":
                        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    elif optim == "Adam":
                        optimizer = Adam(model.parameters(),lr=lr,weight_decay=wd)
                    elif optim == "RMSProp":
                        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=wd)
                    elif optim == "SGD":
                        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.L1Loss()
                    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    

                    loss = train_epochs(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler, early_stop_patience, device, checkpoint_path, best_loss)
                    if loss < best_loss:
                        best_loss = loss 
                        best_params = (lr,wd,optim,batch)



    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_denoised_wavs(model, val_pairs, device, output_dir="../checkpoints/test_outputs")


if __name__ == "__main__":
    main()