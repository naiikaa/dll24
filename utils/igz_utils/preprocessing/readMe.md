# Audio Preprocessing Script

This trims, resamples, and standardizes audio snippets to a fixed duration and sample rate, saving the processed audio as `.wav` files for further processes.

## Features
- **Dataset Loading**: loads audio data from the dataset.
- **Preprocessing**: Converts audio to mono, resamples to a target sample rate, and trims/pads audio to a desired duration.
- **Save and Play**: saves processed audio to a file and plays audio in the terminal.

## Dependencies

To install dependencies, run:
```bash
pip install -r requirements.txt
