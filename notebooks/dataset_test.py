#%%
from datasets import load_dataset

dataset = load_dataset('DBD-research-group/BirdSet','XCM')


#%%
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

#%%
dataset['train'][1000]
#%%
data, samplerate = sf.read(dataset['train'][365]['filepath'])
#%%
dataset['train'][12]
#%%
plt.plot(data)

#%%
from IPython.display import Audio
Audio(data, rate=samplerate)
#%%
import torch
data = []
for i in range(10):
    data.append(torch.randn(1,1,1,64))
#%%
import numpy as np
data = torch.from_numpy(np.array(data))
data.max()
#%%
samplex = torch.tensor(data[:2**17],device='cpu',dtype=torch.float32)
#%%
torch.from_numpy(np.array([torch.tensor(data[:2**17],device='cpu',dtype=torch.float32) for i in range(10)])).min()
#%%
for i in range(400):
    lul,_ = sf.read(dataset['train'][i]['filepath'])
    lulx = torch.tensor(lul[:2**17],device='cpu',dtype=torch.float32)
    print(len(lulx.shape))