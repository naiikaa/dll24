#%%
from datasets import load_dataset

dataset = load_dataset('DBD-research-group/BirdSet','HSN')


#%%
import soundfile as sf
import matplotlib.pyplot as plt

#%%
dataset
#%%
data, samplerate = sf.read(dataset['test_5s'][123]['filepath'])
#%%
dataset['train'][12]
#%%
plt.plot(data)

#%%
from IPython.display import Audio
Audio(data, rate=samplerate)
