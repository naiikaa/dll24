import torch

torch.backends.cudnn.benchmark = True
import numpy as np

torch.set_float32_matmul_precision('medium')
import h5py
from tqdm.auto import tqdm
import dac
from audiotools import AudioSignal
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader


class SnippetDatasetHDF(Dataset):
    def __init__(self, hdf, scaling='minmax'):
        self.num_rows = 0
        self.size = int(3.4 * 24000)
        self.scaling = scaling
        self.data = self.create_data(hdf)

        if scaling == 'standard':
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / self.std

        elif scaling == 'minmax':
            self.min = self.data.min()
            self.max = self.data.max()
            self.data = (self.data - self.min) / (self.max - self.min)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return self.data[idx]

    def create_data(self, hdf):
        data = []
        keys = list(hdf.keys())
        self.num_rows = len(keys)
        for key in tqdm(keys):
            sample = hdf[key]['audio'][:]
            if len(sample) > self.size:
                self.num_rows -= 1
                continue

            if len(sample) < self.size:
                sample = np.pad(sample, (0, self.size - len(sample)), 'constant')

            data.append(sample)

        return torch.tensor(np.array(data)).float()

    def retransform(self, data):
        if self.scaling == 'standard':
            return data * self.std + self.mean
        elif self.scaling == 'minmax':
            return data * (self.max - self.min) + self.min


hdf = h5py.File('../XCM.hdf5', 'r')
dataset = SnippetDatasetHDF(hdf)
hdf.close()

dac_model_path = dac.utils.download(model_type='24kHz')
dac_model = dac.DAC.load(dac_model_path)


def generate_latents(dataset, model):
    latents_list = []
    clen = min(len(dataset), 150)
    for i in range(clen):
        print(i)
        signal = AudioSignal(dataset.retransform(dataset[i]), sample_rate=24000)
        wav_dac = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(wav_dac)
        latents = torch.nn.functional.pad(latents, (0, 1))
        latents_list.append(latents)
    return latents_list


latents_list = generate_latents(dataset, dac_model)
latents_tensor = torch.stack(latents_list)


class LatentDataset(Dataset):
    def __init__(self, latents_tensor):
        self.latents_tensor = latents_tensor

    def __len__(self):
        return len(self.latents_tensor)

    def __getitem__(self, idx):
        return self.latents_tensor[idx]


latent_dataset = LatentDataset(latents_tensor)
torch.save(latent_dataset.latents_tensor, '../models/VQVAE2/latent_dataset.pt')
