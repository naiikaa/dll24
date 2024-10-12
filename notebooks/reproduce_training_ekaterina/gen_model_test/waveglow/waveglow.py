# dep: git clone https://github.com/NVIDIA/waveglow
# from waveglow import glow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os

import torch
import torchaudio
import torch.nn.functional as F


def apply_gain(waveform, gain_db):
    gain = 10 ** (gain_db / 20)
    return waveform * gain


class WaveGlowDataset(Dataset):
    def __init__(self, wav_dir, transform=None, max_length=None):
        self.wav_dir = wav_dir
        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.wav_dir, self.wav_files[idx])
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = apply_gain(waveform, 40)

        if self.transform:
            mel_spec = self.transform(waveform)
        else:
            mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
        mel_spec = self.pad_or_trim(mel_spec, self.max_length)
        waveform = self.pad_or_trim(waveform, self.max_length)
        return mel_spec, waveform

    def pad_or_trim(self, tensor, max_length):
        current_length = tensor.shape[-1]
        if max_length is not None and current_length > max_length:
            return tensor[..., :max_length]
        elif max_length is not None and current_length < max_length:
            pad_amount = max_length - current_length
            return F.pad(tensor, (0, pad_amount), "constant", 0)
        else:
            return tensor


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()
        w_init = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init)
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, z):
        z = torch.clamp(z, min=1e-6, max=1e6)
        if len(z.size()) == 4:
            batch_size, _, num_channels, length = z.size()
        if len(z.size()) == 3:
            batch_size, num_channels, length = z.size()
        z = z.squeeze(1)
        z = self.conv(z)
        return z

    def inverse(self, z):
        z = torch.clamp(z, min=1e-6, max=1e6)
        batch_size, _, num_channels, length = z.size()
        z = z.squeeze(1)
        weight_inv = torch.inverse(self.conv.weight.squeeze()).unsqueeze(2)
        z = torch.nn.functional.conv1d(z, weight_inv)
        return z


class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=3, padding=1)
        )

    def forward(self, z):
        z1, z2 = z.chunk(2, 1)
        z = torch.clamp(z, min=1e-6, max=1e6)
        log_s, t = self.net(z1).chunk(2, 1)
        s = torch.exp(log_s)
        s = torch.clamp(s, min=1e-6, max=1e6)
        t = torch.clamp(s, min=1e-6, max=1e6)
        # log_s = torch.clamp(s, min=1e-6, max=1e6)

        repeat_factor = z2.size(1) // s.size(1)
        s = s.repeat(1, repeat_factor, 1)
        s = s[:, :64, :]
        t = t.repeat(1, repeat_factor, 1)
        t = t[:, :64, :]

        z2 = s * z2 + t
        out = torch.cat([z1, z2], 1)

        # log_det = log_s.sum(dim=1, keepdim=True)
        log_det = log_s.sum(dim=[0, 1, 2])
        # print(f'z1: {z1}, z2: {z2}, log_s: {log_s}, s: {s}, t: {t}')
        return out.unsqueeze(1), log_det

    def inverse(self, z):
        z = torch.clamp(z, min=1e-6, max=1e6)
        z1, z2 = z.chunk(2, 1)
        z1 = torch.clamp(z1, min=1e-6, max=1e6)
        z2 = torch.clamp(z2, min=1e-6, max=1e6)
        log_s, t = self.net(z1).chunk(2, 1)
        s = torch.exp(log_s)
        z2 = (z2 - t) / s
        out = torch.cat([z1, z2], 1)
        return out


class WN(nn.Module):
    def __init__(self, num_channels):
        super(WN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, dilation=2 ** i)
            for i in range(6)
        ])

    def forward(self, x):
        x = x.squeeze(1)
        for idx, conv in enumerate(self.layers):
            x = conv(x)
            x = torch.clamp(x, min=-1e6, max=1e6)
            x = torch.nn.functional.relu(x)
        return x, 0


class WaveGlow(nn.Module):
    def __init__(self, num_channels, num_flows):
        super(WaveGlow, self).__init__()
        self.num_flows = num_flows
        self.flows = nn.ModuleList()

        for _ in range(num_flows):
            self.flows.append(Invertible1x1Conv(num_channels))
            self.flows.append(AffineCoupling(num_channels))
            self.flows.append(WN(num_channels))

    def forward(self, z):
        log_det_jacobian = 0
        z = torch.clamp(z, min=1e-6, max=1e6)

        if len(z.size()) < 4:
            z = z.unsqueeze(1)
        for flow in self.flows:
            if isinstance(flow, Invertible1x1Conv):
                z = flow(z)
                log_det_jacobian += torch.log(torch.abs(torch.det(flow.weight.squeeze(0)))) * z.size(0)
                if torch.isnan(z).any():
                    print("NaN detected after Invertible1x1Conv")
            elif isinstance(flow, AffineCoupling):
                z, log_det = flow(z)
                # avoid getting nans as result
                if not torch.isnan(log_det).any():
                    log_det_jacobian += log_det
                if torch.isnan(z).any():
                    print("NaN detected after AffineCoupling")
            elif isinstance(flow, WN):
                z, log_det = flow(z)
                log_det_jacobian += log_det
                if torch.isnan(z).any():
                    print("NaN detected after WN")
        return z, log_det_jacobian

    def inverse(self, z):
        for flow in reversed(self.flows):
            if isinstance(flow, Invertible1x1Conv):
                z = flow.inverse(z)
            elif isinstance(flow, AffineCoupling):
                z = flow.inverse(z)
            elif isinstance(flow, WN):
                z = flow(z)
        return z


def normalize(spectrograms):
    return (spectrograms - spectrograms.mean()) / (spectrograms.std() + 1e-8)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def train_waveglow(wav_dir, num_epochs=10, batch_size=16, learning_rate=1e-4, device='cuda'):
    # same as in vqvae2 implementation
    num_channels = 64
    mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=num_channels, n_fft=512)
    max_length = 24000
    dataset = WaveGlowDataset(wav_dir=wav_dir, transform=mel_transform, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_flows = 5
    waveglow = WaveGlow(num_channels=num_channels, num_flows=num_flows).to(device)
    optimizer = optim.Adam(waveglow.parameters(), lr=learning_rate)
    initial_lr = 1e-6
    target_lr = 1e-4
    warmup_steps = 1000
    total_steps = 5000

    for training_step in range(total_steps):
        for epoch in range(num_epochs):
            waveglow.train()
            running_loss = 0.0
            torch.nn.utils.clip_grad_norm_(waveglow.parameters(), 1.0)
            waveglow.apply(init_weights)
            # torch.autograd.set_detect_anomaly(True)
            for spectrograms, waveforms in dataloader:
                spectrograms = spectrograms.to(device)
                spectrograms = normalize(spectrograms)
                z, log_det_jacobian = waveglow(spectrograms)
                loss = 0.5 * torch.sum(z ** 2) - log_det_jacobian
                loss = torch.clamp(loss, min=-1e6, max=1e6)

                optimizer.zero_grad()
                loss.backward()

                if training_step < warmup_steps:
                    lr = initial_lr + (target_lr - initial_lr) * (training_step / warmup_steps)
                else:
                    lr = target_lr

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
    torch.save(waveglow.state_dict(), 'waveglow.pth')


train_waveglow('../unpacked_data', num_epochs=10, batch_size=16)
