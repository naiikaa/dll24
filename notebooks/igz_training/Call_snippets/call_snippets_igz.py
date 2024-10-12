import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset

def preprocess_audio_snippets(dataset_name, subset_name, output_dir, desired_length):
    # Create a directory to save snippets
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)
    dataset_train = dataset['train']

    # Process each sample in the training dataset
    for idx, sample in enumerate(dataset_train):
        # Access the audio path
        if 'filepath' in sample:
            audio_path = sample['filepath']
        elif isinstance(sample['audio'], dict) and 'path' in sample['audio']:
            audio_path = sample['audio']['path']
        else:
            print(f"Audio path not found in sample {idx}")
            continue  # Skip this sample if audio path is not found

        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Get the detected events
        detected_events = sample['detected_events']
        if not detected_events:
            continue  # Skip if there are no detected events

        # Process each detected event
        for event_idx, event in enumerate(detected_events):
            # Each event is a list of [start_time, end_time]
            if len(event) != 2:
                print(f"Invalid event format in sample {idx}, event {event_idx}")
                continue
            start_time, end_time = event
            event_duration = end_time - start_time

            # Adjust the start and end times to get a snippet of desired_length
            if event_duration >= desired_length:
                # If the event is longer than desired_length, take the central portion
                center_time = (start_time + end_time) / 2
                start_time_adj = max(0, center_time - desired_length / 2)
                end_time_adj = start_time_adj + desired_length
                if end_time_adj > duration:
                    end_time_adj = duration
                    start_time_adj = end_time_adj - desired_length
            else:
                # If the event is shorter than desired_length, expand the window
                start_time_adj = max(0, start_time - (desired_length - event_duration) / 2)
                end_time_adj = start_time_adj + desired_length
                if end_time_adj > duration:
                    end_time_adj = duration
                    start_time_adj = end_time_adj - desired_length
                if start_time_adj < 0:
                    start_time_adj = 0
                    end_time_adj = desired_length

            # Convert times to sample indices
            start_sample = int(start_time_adj * sr)
            end_sample = int(end_time_adj * sr)

            # Extract the adjusted snippet
            snippet = y[start_sample:end_sample]

            # Ensure the snippet is exactly desired_length seconds
            snippet_length = int(desired_length * sr)
            if len(snippet) < snippet_length:
                # Pad with zeros if necessary
                padding = snippet_length - len(snippet)
                snippet = np.pad(snippet, (0, padding), 'constant')
            elif len(snippet) > snippet_length:
                # Truncate if necessary
                snippet = snippet[:snippet_length]

            # Save the snippet
            ebird_code = sample['ebird_code']
            snippet_filename = f"{ebird_code}_{idx}_{event_idx}.wav"
            snippet_path = os.path.join(output_dir, snippet_filename)
            sf.write(snippet_path, snippet, sr)

        print(f"Processed sample {idx+1}/{len(dataset_train)}")
    
    print('* Call Snippets are split!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess bird audio snippets from a dataset.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--subset_name', type=str, required=True, help='Subset of the dataset (e.g., "HSN")')
    parser.add_argument('--output_dir', type=str, default='bird_snippets', help='Directory to save audio snippets')
    parser.add_argument('--desired_length', type=float, default=5.0, help='Desired snippet length in seconds')
  
    args = parser.parse_args()

    # Use defined function to preprocess audios to find and save call_snippets
    preprocess_audio_snippets(args.dataset_name, args.subset_name, args.output_dir, args.desired_length)
