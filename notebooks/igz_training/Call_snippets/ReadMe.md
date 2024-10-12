* splitting the call snippets and save them as preprocessing:
- load the dataset using `load_dataset` with the configuration `HSN`
- iterate over each sample in `dataset['train']`
- load the audio file using `librosa.load`, preserving the original sampling rate with `sr=None`
- access `sample['detected_events']`, which contains the start and end times of bird calls; use to find call snippets.
- If there are no detected events, we skip the sample.
    For each event, we adjust the start and end times to get a 5-second snippet:
    If the event is longer than 5 seconds, we extract the central 5 seconds.
    If the event is shorter than 5 seconds, we expand the window equally before and after the event to reach 5 seconds, ensuring we don't exceed the audio's duration! :)
- Convert the adjusted start and end times to sample indices.
        Extract the snippet from the audio signal.
        Ensure the snippet is exactly 5 seconds long:
                * Padding with zeros if it's shorter.
                * Truncating if it's longer
- save the snippet as a WAV file using `soundfile.write.`
- The filename includes the ebird_code, sample index, and event index for uniqueness.

* Error Handling:
- The code includes checks for missing audio paths and invalid event formats.
- It prints messages when it skips samples due to missing data.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt


simple command which used to run it before
```bash
python preprocess_bird_calls.py --dataset_name DBD-research-group/BirdSet --subset_name HSN --output_dir bird_snippets --desired_length 5.0
