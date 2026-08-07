from PreProcess import * 
import torch
from RTSEWD import DenoisingRTSEWD
from UNet import DenoisingUNet


def denoise_and_save_wav(model, weight_file_path, in_file_path, out_file_path, device):
    """
        Load a pre-trained model, run denoising on a given input and save to output file.
    """

    model.load_state_dict(torch.load(weight_file_path, map_location=device))
    model.to(device)
    model.eval()

    noisy_waveform, sample_rate = load_audio(in_file_path)
    noisy_waveform, sample_rate = preprocess_audio(noisy_waveform, sample_rate)

    if model.__class__.__name__ == "DenoisingUNet":
        denoise_and_save_wav_unet(model, noisy_waveform, sample_rate, out_file_path, device)
    elif model.__class__.__name__ == "DenoisingRTSEWD":
        denoise_and_save_wav_rtsewd(model, noisy_waveform, sample_rate, out_file_path, device)


def denoise_and_save_wav_unet(model, noisy_waveform, sample_rate, out_file_path, device):
    """
        Run inference on a single waveform using the UNet model and save the denoised output.
    """

    magnitude, phase = compute_stft(noisy_waveform, n_fft=512, hop_length=128)
    magnitude = magnitude.unsqueeze(0).to(device)  
    phase = phase.unsqueeze(0).to(device)

    total_frames = magnitude.shape[-1]
    pad_amount = (16 - total_frames % 16) % 16
    if pad_amount > 0:
        magnitude = torch.nn.functional.pad(magnitude, (0, pad_amount))

    noisy_log = magnitude_to_log(magnitude)
    mask = model(noisy_log)
    mask = mask[..., :total_frames]
    magnitude = magnitude[..., :total_frames]

    estimated_clean = mask * magnitude
    reconstructed = reconstruct_waveform(estimated_clean, phase, n_fft=512, hop_length=128)
    reconstructed = reconstructed.squeeze().cpu().numpy()

    sf.write(out_file_path, reconstructed, samplerate=sample_rate)


def denoise_and_save_wav_rtsewd(model, noisy_waveform, sample_rate, out_file_path, device):
    """
        Run inference on a single waveform using the RTSEWD model and save the denoised output.
    """

    downsample_factor = 2 ** len(model.features)

    noisy_waveform = noisy_waveform.unsqueeze(0).to(device)

    total_len = noisy_waveform.shape[-1]
    pad_amount = (downsample_factor - total_len % downsample_factor) % downsample_factor

    if pad_amount > 0:
        noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_amount))

    estimated_clean = model(noisy_waveform)
    estimated_clean = estimated_clean[..., :total_len]

    sf.write(out_file_path, estimated_clean.squeeze(0).squeeze(0).cpu().numpy(), samplerate=sample_rate)