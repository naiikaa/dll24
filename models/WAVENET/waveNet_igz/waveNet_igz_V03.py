import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import soundfile as sf

# Step 1: Preprocessing In-Memory Snippets

def preprocess_audio_in_memory(example, desired_length=5.0, target_sr=22050):
    # Load the audio file using librosa
    y, sr = librosa.load(example['filepath'], sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Skip if no events are detected
    detected_events = example['detected_events']
    if not detected_events:
        return None

    snippets = []
    for event in detected_events:
        if len(event) != 2:
            continue
        start_time, end_time = event
        event_duration = end_time - start_time

        # Adjust the time window to extract a 5-second snippet
        if event_duration >= desired_length:
            center_time = (start_time + end_time) / 2
            start_time_adj = max(0, center_time - desired_length / 2)
            end_time_adj = start_time_adj + desired_length
            if end_time_adj > duration:
                end_time_adj = duration
                start_time_adj = end_time_adj - desired_length
        else:
            start_time_adj = max(0, start_time - (desired_length - event_duration) / 2)
            end_time_adj = start_time_adj + desired_length
            if end_time_adj > duration:
                end_time_adj = duration
                start_time_adj = end_time_adj - desired_length
            if start_time_adj < 0:
                start_time_adj = 0
                end_time_adj = desired_length

        # Convert start and end times to sample indices and extract the snippet
        start_sample = int(start_time_adj * sr)
        end_sample = int(end_time_adj * sr)
        snippet = y[start_sample:end_sample]

        # Ensure the snippet is exactly the desired length by padding or truncating
        snippet_length = int(desired_length * sr)
        if len(snippet) < snippet_length:
            snippet = np.pad(snippet, (0, snippet_length - len(snippet)), 'constant')
        elif len(snippet) > snippet_length:
            snippet = snippet[:snippet_length]

        # Resample the snippet to the target sample rate
        snippet_resampled = librosa.resample(snippet, orig_sr=sr, target_sr=target_sr)
        
        # Ensure resampled snippet length is correct
        resampled_length = int(desired_length * target_sr)
        if len(snippet_resampled) < resampled_length:
            snippet_resampled = np.pad(snippet_resampled, (0, resampled_length - len(snippet_resampled)), 'constant')
        elif len(snippet_resampled) > resampled_length:
            snippet_resampled = snippet_resampled[:resampled_length]
        
        snippets.append(snippet_resampled)
    
    return snippets

# Load the dataset of bird audio samples
print("Loading dataset...")
dataset = load_dataset('DBD-research-group/BirdSet', 'HSN', trust_remote_code=True)
dataset_train = dataset['train']

# Preprocess each audio sample to extract snippets
print("Preprocessing audio snippets...")
snippets_list = []
for idx, sample in enumerate(dataset_train):
    snippets = preprocess_audio_in_memory(sample)
    if snippets:
        snippets_list.extend(snippets)

# Step 2: Creating a PyTorch Dataset for Training

def mu_law_encoding(x, quantization_channels=256):
    mu = quantization_channels - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((fx + 1) / 2 * mu + 0.5).astype(np.int32)

class BirdAudioDataset(Dataset):
    def __init__(self, snippets, quantization_channels=256):
        self.snippets = snippets
        self.quantization_channels = quantization_channels
        
    def __len__(self):
        return len(self.snippets)
    
    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        snippet = snippet / np.abs(snippet).max()  # Normalize
        y_mu = mu_law_encoding(snippet, self.quantization_channels)
        return torch.from_numpy(y_mu).long().unsqueeze(0)

# Prepare the dataset and dataloader for training
dataset = BirdAudioDataset(snippets_list)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 3: Defining the WaveNet Model

class ResidualBlock(nn.Module):
    def __init__(self, dilation, residual_channels, skip_channels, kernel_size=2):
        super(ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.gate_conv = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        residual_input = x
        filter = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        x = filter * gate
        skip = self.skip_conv(x)
        residual = self.residual_conv(x)
        
        # Remove excess padding and combine residuals
        residual = residual[:, :, :residual_input.size(2)]
        skip = skip[:, :, :residual_input.size(2)]
        
        x = residual + residual_input
        return x, skip

class WaveNet(nn.Module):
    def __init__(self, residual_channels=64, skip_channels=256, quantization_channels=256, dilations=[1, 2, 4, 8, 16, 32, 64], kernel_size=2):
        super(WaveNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=quantization_channels, embedding_dim=residual_channels)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(dilation, residual_channels, skip_channels, kernel_size) for dilation in dilations]
        )
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, quantization_channels, kernel_size=1)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        x = sum(skip_connections)
        x = F.relu(x)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        return x

# Step 4: Training the Model

model = WaveNet()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        x = batch[:, :, :-1]  # Input to the model
        y = batch[:, 0, 1:]  # Target output
        output = model(x)
        min_length = min(output.size(2), y.size(1))
        output = output[:, :, :min_length]
        y = y[:, :min_length]
        output = output.permute(0, 2, 1).reshape(-1, 256)
        y = y.reshape(-1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Step 5: Generating Synthetic Audio

def mu_law_expansion(y_mu, quantization_channels=256):
    mu = quantization_channels - 1
    y = y_mu.astype(np.float32)
    y = y / mu * 2 - 1
    return np.sign(y) * (np.exp(np.abs(y) * np.log1p(mu)) - 1) / mu

def generate(model, initial_input, num_samples, quantization_channels=256):
    model.eval()
    generated = []
    input_sequence = initial_input
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(input_sequence)
            output = output[:, :, -1]
            output = F.softmax(output, dim=1)
            distrib = torch.distributions.Categorical(output)
            sample = distrib.sample()
            generated.append(sample.item())
            sample = sample.unsqueeze(0).unsqueeze(0)
            input_sequence = torch.cat([input_sequence, sample], dim=2)
            if input_sequence.size(2) > 1000:
                input_sequence = input_sequence[:, :, -1000:]
    generated = np.array(generated)
    return mu_law_expansion(generated, quantization_channels)

initial_input = torch.zeros(1, 1, 1).long()
num_samples = 22050 * Here’s the continuation and completion of the comment-refined version:

# Generate 5 seconds of synthetic audio
num_samples = 22050 * 5  # 5 seconds at 22050 Hz sample rate
print("Generating synthetic audio...")
generated_audio = generate(model, initial_input, num_samples)

# Save the generated audio to a file
sf.write('generated_audio_test003.wav', generated_audio, samplerate=22050)
print("Generated audio saved as 'generated_audio_test003.wav'.")
