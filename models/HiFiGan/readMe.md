#### used as foundation:
- `https://github.com/jik876/hifi-gan`
- `https://github.com/chrisdonahue/wavegan`

# HiFi-GAN Audio Preprocessing and Generation

This script preprocesses bird call audio data from the **BirdSet** dataset, trains a **HiFi-GAN** generator model, and generates synthetic audio. The workflow covers everything from loading and preprocessing audio files to training a GAN model and generating audio.

## Key Steps

1. **Audio Preprocessing**:
   - Loads the dataset and extracts bird call snippets.
   - Snippets are trimmed or padded to 5 seconds and resampled to 22,050 Hz.

2. **Dataset Creation**:
   - Preprocessed snippets are organized into a PyTorch dataset, ready for training.

3. **HiFi-GAN Model**:
   - A HiFi-GAN generator model is built using convolutional and transposed convolutional layers.
   - This model is trained to generate realistic audio samples from random noise.

4. **Model Training**:
   - The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer over 50 epochs.

5. **Audio Generation**:
   - Once trained, the model generates synthetic audio from random input, which is saved as a `.wav` file.


## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
