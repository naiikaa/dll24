import sys
sys.path.append('/home/npopkov/dll24')
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
torch.set_float32_matmul_precision('medium')

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import lightning as lt

class LatentDataset(Dataset):
    def __init__(self, h5_file):
        self.shape = np.array(h5_file[list(h5_file.keys())[0]]).shape
        self.data = self.createData(h5_file)
        self.min = self.data.min()
        self.max = self.data.max()
        self.std = self.data.std()
        self.mean = self.data.mean()
        self.transform('normalize')


    def createData(self, h5_file):
        data = []
        for key in h5_file.keys():
        
            sample = np.array(h5_file[key])

            data.append(np.array(sample))
        
        return torch.tensor(np.array(data)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def transform(self, type: str = 'normalize'):
        if type == 'standardize':
            self.data = (self.data - self.mean) / self.std
        elif type == 'normalize':
            self.data = (self.data - self.min) / (self.max - self.min)
        else:
            raise ValueError('Unknown transformation type')
        
    def inverse_transform(self, data):

        return data * (self.max - self.min) + self.min
    
    def unflatten(self, data):
        return data.reshape(self.shape)
    

hdf = h5py.File('256encodesamp.hdf5', 'r')
dataset = LatentDataset(hdf)
hdf.close()





model = Unet(
    dim = 32,
    channels = 1,
    dim_mults = (1, 2, 4,8),
    flash_attn = False,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000    # number of steps
)




class Lightningwrapper(lt.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        batch = batch
        batch = batch.unsqueeze(1)
        loss = self.model(batch)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)

lt_model = Lightningwrapper(diffusion)
dataloader = DataLoader(dataset, batch_size=32,num_workers=30,shuffle=True,)

trainer = lt.Trainer(max_epochs=100)
trainer.fit(lt_model, dataloader)


with torch.no_grad():
    gen_imgs = diffusion.sample(batch_size=32)
gen_imgs = dataset.inverse_transform(gen_imgs)


# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")

#Save sample batch
os.makedirs('samples', exist_ok=True)
torch.save(gen_imgs.data[:25], "samples/sample_batch.pth")