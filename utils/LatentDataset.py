import torch
from torch.utils.data import Dataset
import numpy as np


class LatentDataset(Dataset):
    def __init__(self, h5_file, scaling='normalize'):
        assert scaling in ['standardize', 'normalize', 'none'], f"Expected scaling to be 'standardize', 'normalize', or 'none', got {scaling}"
        
        self.shape = np.array(h5_file[list(h5_file.keys())[0]]).shape
        self.data = self.createData(h5_file)
        
        self.scaling = scaling
        self.min = self.data.min()
        self.max = self.data.max()
        self.std = self.data.std()
        self.mean = self.data.mean()
        self.transform(self.scaling)


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
    
    def transform(self, norm: str = 'normalize'):
        if norm == 'standardize':
            self.data = (self.data - self.mean) / self.std
        elif norm == 'normalize':
            self.data = (self.data - self.min) / (self.max - self.min)
        elif norm == 'none':
            pass
       
        
    def inverse_transform(self, data):
        if self.scaling == 'standardize':
            return data * self.std + self.mean
        elif self.scaling == 'normalize':
            return data * (self.max - self.min) + self.min
        elif self.scaling == 'none':
            return data
      
        
