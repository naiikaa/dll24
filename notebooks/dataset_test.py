#%%
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from datasets import load_dataset
dataset = load_dataset('DBD-research-group/BirdSet','XCM')

#%%
dataset['train'][365]
#%%
x = dataset['train'][365]
detected_events = x['detected_events']
event_clusters = x['event_cluster']
samplerate = 32000
data, _ = sf.read(x['filepath'])
extracted = []

for event, cluster in zip(detected_events, event_clusters):
    
    if cluster != -1:
        start, end = event
        start = int(start * samplerate)
        end = int(end * samplerate)
        extracted.append(data[start:end-1])#

combined = np.concatenate(extracted)
plt.plot(extracted[0])
plt.show()
plt.plot(data)
plt.show()
plt.plot(combined)
plt.show()
print(len(combined))

Audio(combined, rate=samplerate)

#%%
Audio(data, rate=samplerate)
#%%
sf.write('extracted.wav', np.concatenate(extracted), samplerate)
sf.write('original.wav', data, samplerate)

#%%
data, samplerate = sf.read(dataset['train'][365]['filepath'])
#%%
dataset['train'][12]
#%%
plt.plot(data)

#%%
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