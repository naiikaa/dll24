# Utils
## extracting events from BirdSet samples
Raw samples from BirdSet are extracted via the [extraction script](). Running the script will result in a hdf file containing only extracted events from the BirdSet samples. You can also specify following variables:
- `--samplerate`: samplerate in Hz (default: 240000)
- `--save_path` : path the hdf5 file is saved to (default: "./extracted_events.hdf5")
- `--subset` : subset of birdset to use (default: "XCM")
- `--ebird_codes` : list of ebird codes to extract (default: 012)
- `--output_length` : max number of samples to extract (default: 10000)

The structure of the resulting will look somewhat like the following outputs:
```python
hdf = h5py.File(save_path, 'r')

print(hdf.keys())
>> <KeysViewHDF5 ['0008f100-b3dd-42b4-bdf8-b49a87fe2d6d','001f38c5-3f83-4c99-8487-8fa6b1e02d9c', ...]>

print(hdf[list(hdf.keys())[0]].keys())
>> <KeysViewHDF5 ['audio', 'ebird_code', 'ebird_code_multilabel', 'ebird_code_secondary', 'lat', 'local_time', 'long', 'microphone', 'quality', 'soundfile_id', 'source']>
```
Each sample has an unique ID and can later be matched to a certain latent.

## compressing events to latents
Events can now be compressed into latents via [DAC](https://github.com/descriptinc/descript-audio-codec) an audio compression library that takes raw waveforms and compresses them into other representation. The 24kHz model is capable of compressing and decompressing bird audio samples without retraining or any fine tunning see the [DAC test notebook]() to get an idea.
Using the [events_to_latent](dll24/final/utils/events_to_latent.py) script we can take the extracted events hdf5 file from above and create another hdf5 file which now only contains compressed event latents with the corresponding unique ID. Again we can make some specifications:
- `--hdf_path` : path to the hdf5 file that contains events (default: "./extracted_events.hdf5")
- `--scaling` : scaling method (default: "normalize")
    - also available : "standardize", "none"
- `--dac_model_type` : pretrained model types for DAC (default: "24kHz)
- `--save_path` : path to save the latent hdf5 file to (default: "./extraced_latents.hdf5")

