import librosa
import numpy as np
import torch

# Match training settings exactly
TARGET_SR = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MAX_SECONDS = 12.0
MAX_FRAMES = 1200  # ~12s at 10ms hop


def extract_log_mel(
    audio_path,
    target_sr=TARGET_SR,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    max_seconds=MAX_SECONDS,
    max_frames=MAX_FRAMES,
):
    """
    Returns Log-Mel Spectrogram features of shape (time_steps, n_mels),
    matching the training pipeline.
    """
    audio, sr = librosa.load(audio_path, sr=target_sr)

    if max_seconds is not None:
        max_len = int(target_sr * max_seconds)
        if len(audio) > max_len:
            audio = audio[:max_len]

    if len(audio) > 0:
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-9)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=0,
        fmax=target_sr // 2,
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = log_mel.T

    if max_frames is not None and log_mel.shape[0] > max_frames:
        log_mel = log_mel[:max_frames]

    return torch.tensor(log_mel, dtype=torch.float32)
