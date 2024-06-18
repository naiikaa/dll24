#%%
from datasets import load_dataset

dataset = load_dataset('DBD-research-group/BirdSet','HSN')


#%%
import soundfile as sf
import matplotlib.pyplot as plt

data, samplerate = sf.read(dataset['train'][1]['filepath'])
#%%
plt.plot(data)

#%%
from IPython.display import Audio
Audio(data, rate=samplerate)