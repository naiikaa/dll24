import h5py
import numpy as np
import os
import soundfile as sf
from datasets import load_dataset
import uuid
from tqdm.auto import tqdm

wanted_keys = ['audio', 'soundfile_id', 'ebird_code', 'ebird_code_multilabel', 'ebird_code_secondary',
               'lat','long','microphone','source','local_time','quality']

dataset = load_dataset('DBD-research-group/BirdSet','XCM')

hdf = h5py.File('test.hdf5', 'w')

samplerate = 32000

def extract_events(x,hdf):
    detected_events = np.array(x['detected_events'])
    event_clusters = np.array(x['event_cluster'])
    data, _ = sf.read(x['filepath'])
    soundfile_id = str(uuid.uuid4())
        

    for event, cluster in zip(detected_events, event_clusters):
        if len(data.shape) != 1:
            data = data[:,0]

        if cluster != -1:
            start, end = event
            start = int(start * samplerate)
            end = int(end * samplerate)
            x["audio"] = data[start:end-1]
            x["soundfile_id"] = soundfile_id
            try:
                grp = hdf.create_group(str(uuid.uuid4()))
                for key in x.keys():
                    if key in wanted_keys:
                        grp.create_dataset(key, data=x[key])

            except Exception as e:
                print(key)
                print(e)

for x in tqdm(dataset['train']):
    if x['ebird_code'] != 0:
        continue
    
    extract_events(x,hdf)
    if len(hdf.keys()) > 2500:
        break

hdf.close()
