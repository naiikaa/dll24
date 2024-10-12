import argparse
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset
from IPython.display import Audio

# Define a function for preprocessing audio; trims audio files to the same duration and sample rate
def preprocess_audio(example, target_sr=22000, duration_sec=5):
    # Load the audio file using librosa from the 'filepath'
    y, sr = librosa.load(example['filepath'], sr=None)
    
    # Convert to mono if it's stereo
    y_mono = librosa.to_mono(y)
    
    # Resample the audio to the target sample rate
    y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
    
    # Calculate the number of samples for the desired duration
    target_samples = target_sr * duration_sec  # 5 seconds by default = 110,000 samples

    # Trim or pad the audio to the target duration
    y_trimmed = librosa.util.fix_length(y_resampled, size=target_samples)
    
    # Print for debugging purposes
    duration = len(y_trimmed) / target_sr
    print(f"Processed audio duration: {duration} seconds")
    print(f"Number of samples: {len(y_trimmed)}")
    
    # Add the processed audio and sample rate to the example dictionary
    example['processed_audio'] = y_trimmed
    example['sampling_rate'] = target_sr

    return example

# Function to save processed audio to a file
def save_audio(preprocessed, output_filename):
    y_trimmed = preprocessed[0]['processed_audio']
    sampling_rate = preprocessed[0]['sampling_rate']
    sf.write(output_filename, y_trimmed, samplerate=sampling_rate)
    print(f"Audio file '{output_filename}' saved successfully.")

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.subset_name, trust_remote_code=True)

    # Preprocess a selection of the dataset
    preprocessed = dataset['train'].select(range(args.num_samples)).map(
        lambda x: preprocess_audio(x, target_sr=args.sample_rate, duration_sec=args.duration)
    )

    # Optionally save one of the processed samples to a WAV file
    if args.save_audio:
        save_audio(preprocessed, args.output_filename)

    # Optionally play one of the processed audio samples (requires IPython)
    if args.play_audio:
        y_trimmed = preprocessed[0]['processed_audio']
        sampling_rate = preprocessed[0]['sampling_rate']
        display(Audio(y_trimmed, rate=sampling_rate))

if __name__ == "__main__":
    # Define argument parser for command-line parameters
    parser = argparse.ArgumentParser(description="Preprocess audio files and save them to a WAV file.")
    parser.add_argument('--dataset_name', type=str, required=True, help="The name of the dataset (e.g., 'DBD-research-group/BirdSet').")
    parser.add_argument('--subset_name', type=str, required=True, help="The subset of the dataset to load (e.g., 'HSN').")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to preprocess (default: 10).")
    parser.add_argument('--sample_rate', type=int, default=22000, help="Target sample rate for resampling (default: 22000 Hz).")
    parser.add_argument('--duration', type=float, default=5.0, help="Target duration for the audio snippets in seconds (default: 5.0 seconds).")
    parser.add_argument('--save_audio', action='store_true', help="Flag to save the processed audio as a WAV file.")
    parser.add_argument('--output_filename', type=str, default='processed_audio.wav', help="Filename to save the processed audio (default: 'processed_audio.wav').")
    parser.add_argument('--play_audio', action='store_true', help="Flag to play the processed audio in the terminal (requires IPython).")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
