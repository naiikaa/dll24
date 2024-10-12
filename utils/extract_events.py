import h5py
import numpy as np
import os
import soundfile as sf
from datasets import load_dataset
import uuid
from tqdm.auto import tqdm
import librosa
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--samplerate", type=int, default=24000, help="sample rate in Hz")
parser.add_argument("--save_path", type=str, default="./extraced_events.hdf5", help="path to save the hdf5 file")
parser.add_argument("--subset", type=str, default="XCM", help="subset of birdset to use")
parser.add_argument("--ebird_codes", type=list, default=012, help="array of ebird codes to extract")
parser.add_argument("--output_length",type=int, default=10000, help="max number of samples to extract")
opt = parser.parse_args()
print(opt)

wanted_keys = ['audio', 'soundfile_id', 'ebird_code', 'ebird_code_multilabel', 'ebird_code_secondary','lat','long','microphone','source','local_time','quality']

dataset = load_dataset('DBD-research-group/BirdSet',opt.subset)

hdf = h5py.File(opt.save_path, 'w')


def extract_events(x,hdf):
    detected_events = np.array(x['detected_events'])
    event_clusters = np.array(x['event_cluster'])
    data, _ = librosa.load(x['filepath'], sr=opt.samplerate)
    soundfile_id = str(uuid.uuid4())
        

    for event, cluster in zip(detected_events, event_clusters):
        if len(data.shape) != 1:
            data = data[:,0]

        if cluster != -1:
            start, end = event
            start = int(start * opt.samplerate)
            end = int(end * opt.samplerate)
            x["audio"] = data[start:end-1]
            x["soundfile_id"] = soundfile_id
            try:
                grp = hdf.create_group(str(uuid.uuid4()))
                for key in x.keys():
                    if key in wanted_keys:
                        grp.create_dataset(key, data=x[key])
                        if len(hdf.keys()) >= opt.output_length:
                            return

            except Exception as e:
                print(key, e)

for x in tqdm(dataset['train']):
    if str(x['ebird_code']) not in opt.ebird_codes:
        continue
    
    extract_events(x,hdf)
    if len(hdf.keys()) >= opt.output_length:
        print(f"Reached output length")
        break

print(f"Saved {len(hdf.keys())} samples to {opt.save_path}")
hdf.close()

