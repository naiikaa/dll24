import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import soundfile as sf

# Step 1: Preprocessing In-Memory Snippets

def preprocess_audio_in_memory(example, desired_length=5.0, target_sr=22050):
    # Load the audio file using librosa from the 'filepath'
    y, sr = librosa.load(example['filepath'], sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Process each detected event to get a snippet of the desired length
    detected_events = example['detected_events']
    if not detected_events:
        return None  # Skip if there are no detected events

    snippets = []
    for event in detected_events:
        if len(event) != 2:
            continue
        start_time, end_time = event
        event_duration = end_time - start_time

        # Adjust the start and end times to get a 5-second snippet
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

        # Convert times to sample indices
        start_sample = int(start_time_adj * sr)
        end_sample = int(end_time_adj * sr)

        # Extract the adjusted snippet
        snippet = y[start_sample:end_sample]
        snippet_length = int(desired_length * sr)
        
        # Pad or truncate snippet to desired length
        if len(snippet) < snippet_length:
            padding = snippet_length - len(snippet)
            snippet = np.pad(snippet, (0, padding), 'constant')
        elif len(snippet) > snippet_length:
            snippet = snippet[:snippet_length]

        # Resample the snippet to target_sr
        snippet_resampled = librosa.resample(snippet, orig_sr=sr, target_sr=target_sr)
        snippets.append(snippet_resampled)
    
    return snippets

# Load dataset
print("Loading dataset...")
dataset = load_dataset('DBD-research-group/BirdSet', 'HSN', trust_remote_code=True)
dataset_train = dataset['train']

# Preprocess dataset
print("Preprocessing audio snippets...")
snippets_list = []
for idx, sample in enumerate(dataset_train):
    snippets = preprocess_audio_in_memory(sample)
    if snippets:
        snippets_list.extend(snippets)

# Step 2: Creating Dataset for Training

class BirdAudioDataset(Dataset):
    def __init__(self, snippets):
        self.snippets = snippets
        
    def __len__(self):
        return len(self.snippets)
    
    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        snippet = snippet / np.abs(snippet).max()
        return torch.from_numpy(snippet).float().unsqueeze(0)

# Create dataset and dataloader
dataset = BirdAudioDataset(snippets_list)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 3: Defining the HiFi-GAN Generator Model

class HiFiGANGenerator(nn.Module):
    def __init__(self, input_dim=1, ngf=32):
        super(HiFiGANGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, ngf * 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(ngf * 2, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

# Step 4: Training the Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HiFiGANGenerator().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Step 5: Generating Synthetic Audio

def generate(model, initial_input, num_samples):
    model.eval()
    with torch.no_grad():
        generated = model(initial_input)
    return generated.squeeze().cpu().numpy()

initial_input = torch.randn(1, 1, 22050 * 5).to(device)  # 5 seconds of random noise
print("Generating synthetic audio...")
generated_audio = generate(model, initial_input, num_samples=22050 * 5)

# Save the generated audio
sf.write('generated_audio_hifigan_test.wav', generated_audio, samplerate=22050)
print("Generated audio saved as 'generated_audio_hifigan_test.wav'.")
