import h5py
import torchaudio
import random
from datasets import load_dataset
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from IPython.display import Audio
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class OneHot(nn.Module):
    def __init__(self, MU):
        super(OneHot, self).__init__()
        self.MU = MU
        self.ones = torch.sparse.torch.eye(MU).to(device)

    def forward(self, x):
        x = x.to(device)
        x = x.long()  # Ensure x is of type LongTensor
        batch_size, seq_len = x.size()
        x = x.view(-1)

        x_one_hot = self.ones.index_select(0, x)
        x_one_hot = x_one_hot.view(batch_size, seq_len, self.MU)
        x_one_hot = x_one_hot.transpose(1, 2)
        return x_one_hot

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.MU)


class Wavenet(nn.Module):
    def __init__(self, dilation_depth, n_blocks, n_dil_channnels, n_residual_channels, n_skip_channels, n_category,
                 kernel_size, seq_len_segment):
        super(Wavenet, self).__init__()
        self.dilation_depth = dilation_depth
        self.n_blocks = n_blocks
        self.n_dil_channnels = n_dil_channnels
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels
        self.n_category = n_category
        self.kernel_size = kernel_size
        self.One_hot = OneHot(n_category)
        self.seq_len_segment = seq_len_segment
        ###Building the model###
        self.dilations = [2 ** i for i in range(dilation_depth)] * n_blocks

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        ##creating first channels##
        self.input_convs = nn.Conv1d(in_channels=self.n_category, out_channels=self.n_residual_channels, kernel_size=1)
        ###Creating wavenet blocks stacks###
        for d in self.dilations:
            self.filter_convs.append(
                nn.Conv1d(in_channels=n_residual_channels, out_channels=n_dil_channnels, kernel_size=kernel_size,
                          dilation=d))
            self.gate_convs.append(
                nn.Conv1d(in_channels=n_residual_channels, out_channels=n_dil_channnels, kernel_size=kernel_size,
                          dilation=d))
            self.residual_convs.append(
                nn.Conv1d(in_channels=n_dil_channnels, out_channels=n_residual_channels, kernel_size=1))
            self.skip_convs.append(nn.Conv1d(in_channels=n_dil_channnels, out_channels=n_skip_channels, kernel_size=1))
        ##post convoluions
        self.post_conv1 = nn.Conv1d(in_channels=n_skip_channels, out_channels=n_skip_channels, kernel_size=1)
        self.post_conv2 = nn.Conv1d(in_channels=n_skip_channels, out_channels=n_category, kernel_size=1)

    def forward(self, x):
        x = x.to(device)
        x = self.One_hot(x)  # One-hot encoding
        x = self.input_convs(x)  # Input convolution
        skip_con = 0

        for i in range(self.dilation_depth * self.n_blocks):
            dilation = self.dilations[i]
            res = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = self.skip_convs[i](x)
            if isinstance(skip_con, int):
                skip_con = s
            else:
                skip_con = skip_con[:, :, -s.size(2):] + s
            x = self.residual_convs[i](x)
            x = x + res[:, :, dilation * (self.kernel_size - 1):]

        x = torch.relu(skip_con)
        x = torch.relu(self.post_conv1(x))
        x = self.post_conv2(x)

        # Adjust output length to match the target length
        if x.size(2) > self.seq_len_segment:
            x = x[:, :, :self.seq_len_segment]
        elif x.size(2) < self.seq_len_segment:
            padding = self.seq_len_segment - x.size(2)
            x = torch.nn.functional.pad(x, (0, padding))  # Pad

        return x

    def generate(self, seed_input, num_samples=100):
        gen_list = seed_input.squeeze().tolist()
        assert len(gen_list) >= sum(self.dilations) + 1, "Seed input length too short"

        for _ in range(num_samples):
            if len(gen_list) < sum(self.dilations) + 1:
                padding_length = sum(self.dilations) + 1 - len(gen_list)
                gen_list = [0] * padding_length + gen_list

            x = Variable(torch.LongTensor(gen_list[-sum(self.dilations) - 1:]))
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                y = self.forward(x)
                y = y.squeeze(0)  # Remove batch dimension

            _, i = y.max(dim=1)
            gen_list.append(i[-1].item())
        return gen_list


class BirdsetDataset(Dataset):
    def __init__(self, hsn, seq_len_segment, mu):
        # self.hsn = hsn
        self.hsn = self.createData(hsn)
        self.seq_len_segment = seq_len_segment
        self.size = 2 ** 15
        self.mu = mu
        self.data_list = []
        for sample in self.hsn:
            data, _ = preprocess(sample)
            if data.shape[1] >= self.seq_len_segment:
                max_val = torch.max(data)
                min_val = torch.min(data)
                if max_val > torch.abs(min_val):
                    data = data / max_val
                else:
                    data = data / torch.abs(min_val)
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        start = np.random.randint(0, data.shape[1] - self.seq_len_segment)
        ys = data[:, start:start + self.seq_len_segment]
        ys = mulaw_quantize(ys, self.mu)
        ys = ys.squeeze(0)
        return ys.to(device)

    def createData(self, hdf):
        data = []
        keys = list(hdf.keys())

        self.num_rows = len(keys)
        for key in tqdm(keys):
            sample = hdf[key]['audio'][:]
            if len(sample) > self.size:
                self.num_rows -= 1
                continue

            if len(sample) < self.size:
                sample = np.pad(sample, (0, self.size - len(sample)), 'constant')

            data.append(sample)

        return torch.tensor(np.array(data)).float()


class SnippetDatasetHDF(Dataset):
    def __init__(self, hdf, seq_len_segment, mu, scaling='minmax'):
        self.num_rows = 0
        self.size = 2 ** 15
        self.scaling = scaling
        self.hsn = self.createData(hsn)
        self.seq_len_segment = seq_len_segment
        self.size = 2 ** 15
        self.mu = mu
        self.data = []

        for sample in self.hsn:
            data = preprocess(sample)
            # if data.shape[0] >= self.seq_len_segment:
            max_val = torch.max(data)
            min_val = torch.min(data)
            if max_val > torch.abs(min_val):
                data = data / max_val
            else:
                data = data / torch.abs(min_val)
            self.data.append(data)

        self.data = torch.stack(self.data)

        if scaling == 'standard':
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / self.std

        if scaling == 'minmax':
            self.max = torch.max(self.data)
            self.min = torch.min(self.data)
            self.data = (self.data - self.min) / (self.max - self.min)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return self.data[idx]

    def createData(self, hdf):
        data = []
        keys = list(hdf.keys())
        self.num_rows = len(keys)
        for key in tqdm(keys):
            sample = hdf[key]['audio'][:]
            if len(sample) > self.size:
                self.num_rows -= 1
                continue

            if len(sample) < self.size:
                sample = np.pad(sample, (0, self.size - len(sample)), 'constant')

            data.append(sample)

        return torch.tensor(np.array(data)).float()

    def retransform(self, data):
        if self.scaling == 'standard':
            return data * self.std + self.mean
        if self.scaling == 'minmax':
            return data * (self.max - self.min) + self.min


def mulaw_quantize(x, qc):
    assert isinstance(x, torch.Tensor), 'mu_law_encoding expects a Tensor'
    mu = qc - 1
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def inv_mulaw_quantize(x_mu, quantization_channels=256, device='cpu'):
    # Ensure the device is set to the correct value
    device = torch.device(device)
    mu = quantization_channels - 1.  # Calculate mu
    if isinstance(x_mu, np.ndarray):
        x = ((x_mu) / mu) * 2 - 1.
        x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
    elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
        if isinstance(x_mu, (torch.LongTensor, torch.cuda.LongTensor)):
            x_mu = x_mu.float()

        x_mu = x_mu.to(device)
        mu_tensor = torch.FloatTensor([mu]).to(device)
        x = ((x_mu) / mu_tensor) * 2 - 1.
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu_tensor)) - 1.) / mu_tensor
    return x


def data_generation(data, fr, seq_len_segment, mu, device=device):
    max = torch.max(data)
    min = torch.min(data)
    comparison = max > torch.abs(min)
    if torch.all(comparison):
        data = torch.div(data, max)
    else:
        # abs_min_vals = torch.abs(min)
        data = torch.div(data, abs(min))
    while True:
        start = np.random.randint(0, data.shape[1] - seq_len_segment)
        ys = data[:, start:start + seq_len_segment]
        ys = mulaw_quantize(ys, mu)
        ys = ys.squeeze(0)
        yield ys.to(device)


def preprocess(batch):
    print(batch)
    # audio, _ = torchaudio.load(batch['filepath'])
    if isinstance(batch, torch.Tensor):
        audio = batch
    else:
        audio, _ = torchaudio.load(batch)
    if audio.shape[0] > 1:
        audio = audio[0]
    if len(audio.shape) > 1:
        audio = audio.mean(dim=0)
    return audio.unsqueeze(0)


# hsn = load_dataset('DBD-research-group/BirdSet', 'HSN')
hsn = h5py.File('./test_24k.hdf5', 'r')
subset_percentage = 0.5
seq_len_segment = 4000
mu = 128
batch_size = 8
# dataset = BirdsetDataset(hsn, seq_len_segment, mu)
# dataset = SnippetDatasetHDF(hsn)
dataset = SnippetDatasetHDF(hsn, seq_len_segment, mu)
# hsn.close()
# subset_indices = random.sample(range(len(hsn['train'])), int(len(hsn['train']) * subset_percentage))
# hsn = hsn['train'].select(subset_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dilation_depth = 10
n_blocks = 2
n_dilation_channels = 24
n_residual_channels = 24
n_skip_channels = 128
n_category = mu
kernel_size = 2
model = Wavenet(dilation_depth, n_blocks, n_dilation_channels, n_residual_channels, n_skip_channels, n_category,
                kernel_size, seq_len_segment=seq_len_segment)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 35
model.train()
for epoch in range(epochs):
    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs = outputs.permute(0, 2, 1)

        batch_size, seq_len, num_classes = outputs.size()
        outputs = outputs.contiguous().view(-1, num_classes)
        targets = inputs.contiguous().view(-1)

        targets = targets.long()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item() / 10:.4f}')

print('saving model')
torch.save(model.state_dict(), 'wavenet_model.pth')
