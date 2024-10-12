import os

from notebooks.reproduce_training_ekaterina import BirdSetDataModule, DatasetConfig
from datasets import load_dataset
import lightning as lt
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import torch.nn.functional as F
from tqdm.auto import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# initiate the data module
os.environ['HYDRA_FULL_ERROR'] = '1'
import torch
import argparse

try:
    from networks.vqvae2 import VQVAE
except ImportError:
    from models.VQVAE2.ECOGEN_V.vqvae2 import VQVAE

parser = argparse.ArgumentParser(description="Train VQVAE on BirdSet dataset.")
parser.add_argument('--data_paths', type=str, default='./data_birdset/HSN/*', help='Paths to data samples.')
parser.add_argument("--out_folder", type=str, default='.', help="Output folder for generated samples.")
parser.add_argument('--augmentations', default="noise")
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--device', default='cpu')
parser.add_argument('--model_path', type=str, default="epoch=5-step=30005.ckpt")


class AudioDataset(Dataset):
    def __init__(self, dataset, key='train'):
        self.key = key
        self.num_rows = 0
        self.target_shape = (1, 1, 64, 64)
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
            if self.num_rows > 100:
                break
            file_path = row['filepath']
            sample, samplerate = sf.read(file_path)
            if len(sample.shape) != 1:
                sample = sample[:, 0]
            if len(sample) < 2 ** 16:
                continue
            if row['quality'] in ['B', 'C']:
                continue

            # Convert to tensor
            samplex = torch.tensor(sample[:2 ** 16], dtype=torch.float32)
            samplex = samplex.squeeze().view(1, 256, 256)
            self.num_rows += 1
            data.append(samplex)

        return torch.stack(data)


class Lightningwrapper(lt.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred, latent_loss = self.model(batch)
        recon_loss = F.mse_loss(pred, batch.squeeze())
        loss = latent_loss + recon_loss
        self.log('train_loss', loss)
        return loss

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, quant_t, quant_b):
        return self.model.decode(quant_t, quant_b)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)


def main() -> None:
    args = parser.parse_args()
    model = VQVAE(in_channel=1)
    # if args.model_path:
    #    model = load_model(model, args.model_path, device=args.device)

    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir='../../../notebooks/reproduce_training_ekaterina/gen_model_test/data_birdset/HSN',
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

    hsn = load_dataset('DBD-research-group/BirdSet', 'HSN')
    dataset = AudioDataset(hsn)

    lt_model = Lightningwrapper(model)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=30)

    # Train model with audio waveforms
    trainer = lt.Trainer(max_epochs=30)
    trainer.fit(lt_model, dataloader)
    torch.save(lt_model.state_dict(),
               '../../../notebooks/reproduce_training_ekaterina/gen_model_test/checkpoints/epoch=20.ckpt')


if __name__ == "__main__":
    main()
