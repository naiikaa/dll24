# WaveNet


## Overview

Implements a WaveNet model for generating synthetic audio using PyTorch. It processes bird audio samples from the BirdSet dataset, encodes them using mu-law encoding, and generates new audio snippets using the trained WaveNet model.

### Model Flow

#### Step 1: Audio Preprocessing
1. Audio data is loaded using `librosa`.
2. Each detected event in the audio file is processed:
    - The start and end times are adjusted to extract 5-second snippets.
    - The snippets are padded or truncated to ensure they are exactly 5 seconds long.
    - Snippets are resampled to the target sample rate (e.g., 22050 Hz).
3. The processed audio snippets are returned for further processing.

#### Step 2: Dataset Preparation
1. The processed audio snippets are mu-law encoded to compress the audio values.
2. Custom PyTorch `Dataset` is created to store the encoded snippets.
3. PyTorch `DataLoader` is also used to batch and shuffle the dataset for training.

#### Step 3: WaveNet Model Architecture
1. Input audio snippets(which are also mu-law encoded) are passed through an embedding layer.
2. The input is processed through several residual blocks:
    - Each residual block consists of:
        - `dilated convolution` with tanh activation (as filter).
        - `parallel convolution` with sigmoid activation (as gate).
        - Element-wise multiplication of the filter and gate outputs.
        - `1x1 convolution` to generate the residual output, which is added back to the input (residual connection).
        - `skip connection` is also generated and passed to the next layer.
3. All skip connections are summed, passed through ReLU activations, and processed by means of two 1x1 convolutions to produce the final output.

#### Step 4: Model Training
1. **Loss Function**: uses CrossEntropyLoss for training.
2. **Training Loop**:
    - For each batch, the input is passed through the WaveNet model.
    - The loss between the model output and target is computed.
    - Backpropagation is performed, and the model weights are optimized using the AdamW optimizer.
3. The training loop is repeated for multiple epochs.

#### Step 5: Generating Synthetic Audio
1. **Autoregressive Generation**:
    - The trained WaveNet model generates audio samples(one at a time).
    - Each sample is *conditioned* on the previously generated samples.
    - A categorical distribution is used to sample the next output.
    - The generated *mu-law encoded outputs* are converted back into a waveform.
2. **Output**:
    - The generated audio is saved to a file.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
