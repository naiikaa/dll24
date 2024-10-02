import h5py
import numpy as np
import os
import soundfile as sf
import torch
from datasets import load_dataset
import uuid
from tqdm.auto import tqdm
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim, MelSpectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"

wanted_keys = ['audio', 'soundfile_id', 'ebird_code', 'ebird_code_multilabel', 'ebird_code_secondary',
               'lat', 'long', 'microphone', 'source', 'local_time', 'quality']

dataset = load_dataset('DBD-research-group/BirdSet', 'HSN')

hdf = h5py.File('HSN.hdf5', 'w')
samplerate = 24000

def extract_events(x, hdf):
    print('extracting events')
    detected_events = np.array(x['detected_events'])
    event_clusters = np.array(x['event_cluster'])
    data, _ = librosa.load(x['filepath'], sr=samplerate)
    soundfile_id = str(uuid.uuid4())

    for event, cluster in zip(detected_events, event_clusters):
        if len(data.shape) != 1:
            data = data[:, 0]

        if cluster != -1:
            start, end = event
            start = int(start * samplerate)
            end = int(end * samplerate)
            x["audio"] = data[start:end - 1]
            x["soundfile_id"] = soundfile_id
            try:
                grp = hdf.create_group(str(uuid.uuid4()))
                for key in x.keys():
                    if key in wanted_keys:
                        grp.create_dataset(key, data=x[key])

            except Exception as e:
                print(key)
                print(e)


for x in tqdm(dataset['train']):
    if x['ebird_code'] != 0:
        continue

    extract_events(x, hdf)
    if len(hdf.keys()) > 10000:
        break

hdf.close()


def unpack_hdf5(hdf5_file, output_dir, save_as_wav=True, sampling_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(hdf5_file, 'r') as f:
        def recursively_extract(name, obj):
            print(name)
            if 'audio' in name:
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if save_as_wav:
                        output_path = os.path.join(output_dir, f"{name.replace('/', '_')}.wav")
                        sf.write(output_path, data, samplerate=sampling_rate)
                        print(f"Saved {name} as WAV file at {output_path}")
                    else:
                        output_path = os.path.join(output_dir, f"{name.replace('/', '_')}.npy")
                        np.save(output_path, data)
                        print(f"Saved {name} as numpy array at {output_path}")

        f.visititems(recursively_extract)


hdf5_file = './HSN.hdf5'
output_dir = './unpacked_data'
unpack_hdf5(hdf5_file, output_dir, save_as_wav=True, sampling_rate=16000)

def preprocess_wav(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    n_fft = 2048  # higher FFT size for better frequency resolution
    hop_length = 32  # smaller val for better temporal resolution
    n_mels = 256  # higher val for better frequency resolution

    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spectrogram = mel_transform(waveform)

    return mel_spectrogram

def apply_gain(waveform, gain_db):
    gain = 10 ** (gain_db / 20)
    return waveform * gain

def compress_dynamic_range(waveform, threshold=-20.0, ratio=5.0):
    db_waveform = 20 * torch.log10(waveform.abs() + 1e-10)
    db_waveform = torch.where(db_waveform > threshold,
                              threshold + (db_waveform - threshold) / ratio,
                              db_waveform)
    compressed_waveform = 10 ** (db_waveform / 20)
    return compressed_waveform


# test
file_path = '../unpacked_data/0aee9bcf-8656-40b6-b915-35ec63d85ba8_audio.wav'
mel_spectrogram = preprocess_wav(file_path)
print(mel_spectrogram.shape)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_embeddings, embedding_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = self.dropout(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = self.dropout(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / n_embeddings, 1 / n_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = torch.index_select(self.embeddings.weight, 0, encoding_indices.view(-1))
        quantized = quantized.view(inputs.shape)

        # Commitment Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices


class VQVAE2(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, n_embeddings=512, embedding_dim=64):
        super(VQVAE2, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, n_embeddings, embedding_dim)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        x_recon = F.interpolate(x_recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x_recon, vq_loss


from torch.utils.data import Dataset, DataLoader
import os


class WavDataset(Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        mel_spectrogram = preprocess_wav(file_path)
        return mel_spectrogram


def pad_collate(batch):
    max_len = max([x.shape[-1] for x in batch])
    batch = [F.pad(x, (0, max_len - x.shape[-1])) for x in batch]
    return torch.stack(batch)


dataset = WavDataset(file_dir='./unpacked_data')
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)


def train(model, data_loader, epochs=150, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            mel_spectrogram = batch.to(device)

            recon, vq_loss = model(mel_spectrogram)
            recon_loss = criterion(recon, mel_spectrogram)
            loss = recon_loss + 0.5 * vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')


model = VQVAE2(in_channels=1).to(device)
train(model, data_loader, epochs=25)


def generate_samples(model, num_samples=1, spectrogram_shape=(128, 80), device=device):
    model.eval()
    with torch.no_grad():
        latent_code = torch.randn(num_samples, model.vq.embedding_dim, spectrogram_shape[0] // 4,
                                  spectrogram_shape[1] // 4).to(device)
        generated_spectrogram = model.decoder(latent_code)
        generated_spectrogram = generated_spectrogram.squeeze(1)
        return generated_spectrogram


def spectrogram_to_waveform(mel_spectrogram, sample_rate=16000, n_fft=2048, n_mels=256):
    inverse_mel_scale = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels).to(mel_spectrogram.device)
    linear_spectrogram = inverse_mel_scale(mel_spectrogram)
    # griffin_lim = GriffinLim(n_fft=n_fft).to(mel_spectrogram.device)
    griffin_lim = GriffinLim(n_fft=n_fft, n_iter=100).to(mel_spectrogram.device)  # More iterations
    waveform = griffin_lim(linear_spectrogram)
    return waveform


def save_generated_audio(model, file_name="generated_sample.wav", device="cuda"):
    generated_mel_spectrogram = generate_samples(model, num_samples=1, device=device)
    generated_waveform = spectrogram_to_waveform(generated_mel_spectrogram, sample_rate=16000)
    generated_waveform = generated_waveform.cpu()
    generated_waveform = generated_waveform.unsqueeze(0) if generated_waveform.ndim == 1 else generated_waveform
    # output results are too quiet
    generated_waveform = apply_gain(generated_waveform, 40)
    torchaudio.save(file_name, generated_waveform, 16000)


save_generated_audio(model, file_name="generated_sample.wav", device=device)