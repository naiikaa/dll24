from collections import OrderedDict
from copy import deepcopy
import random
import os
import pyarrow as pa
import pyarrow.dataset as ds
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
import pyarrow as pa
import pyarrow.parquet as pq
from notebooks.reproduce_training_ekaterina import BirdSetDataModule, DatasetConfig
import torch.optim as optim
import warnings
from datasets import load_dataset
import lightning as lt
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# initiate the data module
os.environ['HYDRA_FULL_ERROR'] = '1'
import torch
import argparse
import glob
import librosa

try:
    from networks.vqvae2 import VQVAE
except ImportError:
    from vqvae2 import VQVAE

import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="Train VQVAE on BirdSet dataset.")
parser.add_argument('--data_paths', type=str, default='./data_birdset/HSN/*', help='Paths to data samples.')
parser.add_argument("--out_folder", type=str, default='.', help="Output folder for generated samples.")
parser.add_argument('--augmentations', default="noise")
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--device', default='cpu')
parser.add_argument('--model_path', type=str, default="epoch=5-step=30005.ckpt")

from tqdm.auto import tqdm


class AudioDataset(Dataset):
    def __init__(self, dataset, key='train'):
        self.key = key
        self.num_rows = 0
        self.target_shape = (1, 1, 64, 64)  # Example target shape (channels, height, width)
        self.data = self.createData(dataset)

        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return self.data[idx]

    def createData(self, dataset):
        data = []

        for row in tqdm(dataset[self.key]):
            if self.num_rows > 2000:
                break
            file_path = row['filepath']
            sample, samplerate = sf.read(file_path)
            if len(sample) < 2 ** 16:
                continue
            if row['quality'] in ['B', 'C']:
                continue

            # Convert to tensor
            samplex = torch.tensor(sample[:2 ** 16], dtype=torch.float32)

            # Ensure single channel input
            if len(samplex.shape) > 1:
                samplex = samplex.mean(dim=1, keepdim=True)  # Convert to 1 channel if multi-channel
            else:
                samplex = samplex.unsqueeze(0)  # Add channel dimension if needed

            # Ensure 4D tensor
            if len(samplex.shape) == 2:
                samplex = samplex.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Handle tensors with varying shapes
            if len(samplex.shape) == 4:
                # Check if resizing is needed
                if samplex.shape[2] != self.target_shape[2] or samplex.shape[3] != self.target_shape[3]:
                    # Resize to the target shape using interpolation
                    samplex = nn.functional.interpolate(samplex, size=(self.target_shape[2], self.target_shape[3]),
                                                        mode='bilinear', align_corners=False)
            else:
                raise ValueError(f"Unexpected tensor shape {samplex.shape}")

            self.num_rows += 1
            data.append(samplex)

        return torch.stack(data)


class VQVAE(nn.Module):
    def __init__(self, in_channel=1):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Add other layers
        )


def main() -> None:
    args = parser.parse_args()
    model = VQVAE(in_channel=1)
    # if args.model_path:
    #    model = load_model(model, args.model_path, device=args.device)

    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir='./data_birdset/HSN',
            dataset_name='HSN',
            hf_path='DBD-research-group/BirdSet',
            hf_name='HSN',
            n_workers=3,
            val_split=0.2,
            task="multilabel",
            classlimit=500,
            eventlimit=5,
            sampling_rate=32000,
        ),
    )

    class Lightningwrapper(lt.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            loss = self.model(batch.unsqueeze(1))
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    hsn = load_dataset('DBD-research-group/BirdSet', 'HSN')
    dataset = AudioDataset(hsn)

    lt_model = Lightningwrapper(model)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=30)

    # Train model with audio waveforms
    trainer = lt.Trainer(max_epochs=40)
    trainer.fit(lt_model, dataloader)


if __name__ == "__main__":
    main()
