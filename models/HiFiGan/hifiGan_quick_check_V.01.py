import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import soundfile as sf

# Preprocess audio by trimming to a desired length and resampling
def preprocess_audio_in_memory(example, desired_length=2.0, target_sr=11025):
    y, sr = librosa.load(example['filepath'], sr=None)  # Load audio file
    duration = librosa.get_duration(y=y, sr=sr)  # Get audio duration
    
    detected_events = example['detected_events']  # Get detected events
    if not detected_events:
        return None  # Skip if no events

    snippets = []
    for event in detected_events:
        if len(event) != 2:
            continue
        start_time, end_time = event
        event_duration = end_time - start_time

        # Adjust event to fit desired length
        if event_duration >= desired_length:
            center_time = (start_time + end_time) / 2
            start_time_adj = max(0, center_time - desired_length / 2)
        else:
            start_time_adj = max(0, start_time - (desired_length - event_duration) / 2)

        end_time_adj = start_time_adj + desired_length
        if end_time_adj > duration:
            end_time_adj = duration
            start_time_adj = end_time_adj - desired_length

        # Convert time to sample indices
        start_sample = int(start_time_adj * sr)
        end_sample = int(end_time_adj * sr)
        
        # Extract and resample snippet
        snippet = y[start_sample:end_sample]
        snippet_resampled = librosa.resample(snippet, orig_sr=sr, target_sr=target_sr)
        snippets.append(snippet_resampled)
    
    return snippets

# Load and preprocess a dataset, limiting the number of samples
def load_and_preprocess_dataset(limit=200):
    dataset = load_dataset('DBD-research-group/BirdSet', 'HSN', trust_remote_code=True)
    dataset_train = dataset['train'].select(range(limit))
    
    snippets_list = []
    for sample in dataset_train:
        snippets = preprocess_audio_in_memory(sample)
        if snippets:
            snippets_list.extend(snippets)
    
    return snippets_list

# Custom dataset for audio snippets
class BirdAudioDataset(Dataset):
    def __init__(self, snippets):
        self.snippets = snippets
        
    def __len__(self):
        return len(self.snippets)
    
    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        snippet = snippet / np.abs(snippet).max()  # Normalize audio
        return torch.from_numpy(snippet).float().unsqueeze(0)

# HiFi-GAN Generator model definition
class HiFiGANGenerator(nn.Module):
    def __init__(self, input_dim=1, ngf=16):
        super(HiFiGANGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, ngf * 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(ngf * 2, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

# Training the HiFi-GAN model on the preprocessed audio snippets
def train_hifigan(snippets_list, num_epochs=1, batch_size=2):
    dataset = BirdAudioDataset(snippets_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiFiGANGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

    return model

# Generate synthetic audio using the trained HiFi-GAN model
def generate_audio(model, num_samples=11025 * 2):
    initial_input = torch.randn(1, 1, num_samples).to(model.device)
    model.eval()
    with torch.no_grad():
        generated = model(initial_input)
    return generated.squeeze().cpu().numpy()

# Save the generated audio to a file
def save_audio_to_file(generated_audio, filename='generated_HiFiGan_quick_version.wav', samplerate=11025):
    sf.write(filename, generated_audio, samplerate=samplerate)
    print(f"Audio saved as {filename}.")

# Main function to load data, train model, generate audio, and save the result
def main():
    snippets_list = load_and_preprocess_dataset(limit=200)
    model = train_hifigan(snippets_list, num_epochs=1)
    generated_audio = generate_audio(model)
    save_audio_to_file(generated_audio)
    print("Audio generated and saved to disk.")

if __name__ == "__main__":
    main()
