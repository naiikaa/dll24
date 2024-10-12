from EventDataset import EventDataset
import torch
import h5py
import argparse
import dac
from audiotools import AudioSignal
import torchaudio
import numpy as np
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--hdf_path", type=str, default="./extraced_events.hdf5", help="path to the hdf5 file")
parser.add_argument("--scaling", type=str, default="normalize", help="scaling method")
parser.add_argument("--dac_model_type", type=str, default="24kHz", help="model type for DAC")
parser.add_argument("--save_path", type=str, default="./extraced_latents.hdf5", help="path to save the latent hdf5 file")
opt = parser.parse_args()
print(opt)

hdf = h5py.File(opt.hdf_path, 'r')
dataset = EventDataset(hdf, opt.scaling)
hdf.close()

print(f"Loaded {len(dataset)} samples from {opt.hdf_path}")
print(f"Dataset shape: {dataset.data.shape}")

audiocompressor_path = dac.utils.download(model_type=opt.dac_model_type)
audiocompressor = dac.DAC.load(audiocompressor_path)

latent_hdf = h5py.File(opt.save_path, 'w')

for uuid, sample in tqdm(zip(dataset.uuids, dataset.data)):
    sample = dataset.inverse_transform(sample)
    sample = AudioSignal(sample.numpy(), sample_rate=24000)
    
    wav_dac = audiocompressor.preprocess(sample.audio_data, sample_rate=24000)
    _,_, latent, _, _  = audiocompressor.encode(wav_dac)
    latent = torch.nn.functional.pad(latent,(0,1))
    latent_hdf.create_dataset(uuid, data=latent.squeeze().detach().cpu().numpy())
    
latent_hdf.close()
print(f"Transformed {len(dataset)} samples into latents and saved to {opt.save_path}")
    