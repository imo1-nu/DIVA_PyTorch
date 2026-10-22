"""Pretrained DiffWave plus the independently trained conditioner upsampler."""
import math
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vendor.model import DiffWave
from vendor.params import AttrDict, params

SR, HOP, N_MELS = 22050, 256, 80


class MelSpectrogram(nn.Module):
    """Published SpeechBrain checkpoint recipe: magnitude Slaney mel, ln.

    The model card and LJSpeech training YAML call HifiGAN.mel_spectrogram
    with compression=True. DiffWave.diffwave_mel_spectrogram's [0,1] scaling
    is a different function and must not be substituted for these weights.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(1024))
        self.register_buffer('filters', torch.from_numpy(librosa.filters.mel(
            sr=SR, n_fft=1024, n_mels=N_MELS, fmin=0, fmax=8000,
            htk=False, norm='slaney')))

    def forward(self, audio):
        spectrum = torch.stft(audio, n_fft=1024, hop_length=HOP,
                              win_length=1024, window=self.window, center=True,
                              pad_mode='reflect', return_complex=True).abs()
        mel = torch.matmul(self.filters, spectrum)
        return torch.log(mel.clamp_min(1e-5))


def load_vocoder(checkpoint, device='cpu'):
    """Map SpeechBrain's wrapper names, then require a complete strict load.

    SpeechBrain Conv1d uses reflection padding; LMNT uses zero padding. Set
    the padding explicitly rather than silently changing the pretrained model.
    """
    model = DiffWave(AttrDict(params))
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    mapped = {key.replace('.conv.', '.').replace('.w.', '.'): value
              for key, value in state.items()}
    model.load_state_dict(mapped, strict=True)
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            module.padding_mode = 'reflect'
    model.to(device).eval().requires_grad_(False)
    # Stabilize TorchScript's profiling/warmup before comparing inference
    # paths or using the fixed sampling seed. No parameters are changed.
    with torch.no_grad():
        for _ in range(4):
            model.diffusion_embedding(torch.tensor([0.0], device=device))
    return model


class ConditionerUpsampler(nn.Module):
    """Literal channel sequence from Zhang et al. Fig. 2 (2021).

    Eight 5x5 encoder convolutions have outputs 1,4,8,16,64,64,64,64.
    Eight decoder convolutions include four 4x transposed convolutions.
    Padding, and interpretation of the ambiguous input/layer count, are
    explicitly implementation choices: the literal diagram gives 16 learned
    convolutions whereas the reference text says 15. No author weights exist.
    """
    def __init__(self):
        super().__init__()
        layers = []
        current = 1

        def convolution(channels):
            nonlocal current
            layers.extend([nn.Conv2d(current, channels, 5, padding=2),
                           nn.BatchNorm2d(channels), nn.LeakyReLU(0.4)])
            current = channels

        def upsample():
            layers.extend([nn.ConvTranspose2d(current, current, (3, 8),
                                             stride=(1, 4), padding=(1, 2)),
                           nn.BatchNorm2d(current), nn.LeakyReLU(0.4)])

        for channels in (1, 4, 8, 16, 64, 64, 64, 64):
            convolution(channels)
        upsample()
        convolution(16)
        upsample()
        convolution(8)
        upsample()
        convolution(4)
        convolution(1)
        upsample()
        self.network = nn.Sequential(*layers)

    def forward(self, mel):
        return self.network(mel.unsqueeze(1)).squeeze(1)


def predict_noise(model, audio, step, conditioner):
    # Same residual computation as the vendored forward; conditioner is
    # computed once outside the reverse process, including the deep CNN.
    x = F.relu(model.input_projection(audio.unsqueeze(1)))
    embedding = model.diffusion_embedding(step)
    skip = None
    for layer in model.residual_layers:
        x, connection = layer(x, embedding, conditioner)
        skip = connection if skip is None else skip + connection
    x = skip / math.sqrt(len(model.residual_layers))
    return model.output_projection(F.relu(model.skip_projection(x))).squeeze(1)


@torch.inference_mode()
def synthesize(model, mel, upsampler=None, fast=False, seed=0):
    """LMNT reverse process with fixed random seed and cached conditioning."""
    device = next(model.parameters()).device
    mel = mel.to(device)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    conditioner = (model.spectrogram_upsampler if upsampler is None else upsampler)(mel)
    train_beta = np.asarray(model.params.noise_schedule)
    beta = np.asarray(model.params.inference_noise_schedule) if fast else train_beta
    train_alpha_cum = np.cumprod(1 - train_beta)
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    steps = []
    for target in alpha_cum:
        for index in range(len(train_beta) - 1):
            if train_alpha_cum[index+1] <= target <= train_alpha_cum[index]:
                fraction = ((np.sqrt(train_alpha_cum[index]) - np.sqrt(target)) /
                            (np.sqrt(train_alpha_cum[index]) - np.sqrt(train_alpha_cum[index+1])))
                steps.append(index + fraction)
                break
        else:
            raise ValueError('Inference noise schedule is outside the trained schedule')
    generator = torch.Generator(device=device).manual_seed(seed)
    audio = torch.randn(mel.shape[0], conditioner.shape[-1], generator=generator, device=device)
    for index in range(len(beta)-1, -1, -1):
        step = torch.tensor([steps[index]], device=device, dtype=torch.float32)
        noise = predict_noise(model, audio, step, conditioner)
        audio = (audio - beta[index] / np.sqrt(1-alpha_cum[index]) * noise) / np.sqrt(alpha[index])
        if index:
            sigma = np.sqrt((1-alpha_cum[index-1]) / (1-alpha_cum[index]) * beta[index])
            audio += sigma * torch.randn(audio.shape, generator=generator, device=device)
        audio.clamp_(-1, 1)
    return audio
