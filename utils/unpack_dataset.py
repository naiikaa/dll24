import os
import h5py
import numpy as np
import soundfile as sf

def unpack_hdf5(hdf5_file, output_dir, save_as_wav=True, sampling_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(hdf5_file, 'r') as f:
        def recursively_extract(name, obj):
            print(name)
            if 'audio' in name:
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if save_as_wav:
                        output_path = os.path.join(output_dir, f"{name.replace('/', '_')}.wav")
                        sf.write(output_path, data, samplerate=sampling_rate)
                        # print(f"Saved {name} as WAV file at {output_path}")
                    else:
                        output_path = os.path.join(output_dir, f"{name.replace('/', '_')}.npy")
                        np.save(output_path, data)
                        # print(f"Saved {name} as numpy array at {output_path}")
        f.visititems(recursively_extract)

hdf5_file = '../test_24k.hdf5'
output_dir = './unpacked_data'
unpack_hdf5(hdf5_file, output_dir, save_as_wav=True, sampling_rate=16000)
