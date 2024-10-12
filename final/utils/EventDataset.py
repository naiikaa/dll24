
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import torch
import h5py

class EventDataset(Dataset):
    def __init__(self, hdf, scaling='normalize'):
        assert scaling in ['standardize', 'normalize', 'none'], f"Expected scaling to be 'standardize', 'normalize', or 'none', got {scaling}"
        
        self.data = []
        self.scaling = scaling
        self.size = int(3.4 * 24000) # 3.4 seconds of audio on 24kHz to create the right size of latent later
        self.uuids = []
        self.createData(hdf)
        
        self.min = self.data.min()
        self.max = self.data.max()
        self.std = self.data.std()
        self.mean = self.data.mean()
        
        self.transform(self.scaling)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def createData(self, hdf):
        for key in tqdm(hdf.keys()):
            sample = hdf[key]['audio'][:]
            
            if len(sample)>self.size:
                continue
            
            if len(sample)<self.size:
                sample = np.pad(sample, (0,self.size-len(sample)), 'constant')
                
            self.data.append(np.array(sample))
            self.uuids.append(key)
            
        self.data = torch.tensor(np.array(self.data)).float()
    
    def transform(self, scaling: str = 'normalize'):
        if scaling == 'standardize':
            self.data = (self.data - self.mean) / self.std
        elif scaling == 'normalize':
            self.data = (self.data - self.min) / (self.max - self.min)
        elif scaling == 'none':
            pass

    
    def inverse_transform(self, data):
        if self.scaling == 'standardize':
            return data * self.std + self.mean
        elif self.scaling == 'normalize':
            return data * (self.max - self.min) + self.min
