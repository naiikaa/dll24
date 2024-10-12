import h5py
import torchaudio
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim
from tqdm.auto import tqdm

# modified version of WAVENET model from:
# https://github.com/prantoshdas/Pytorch_Wavenet/blob/main/Wavent_notebook/Final_wavenet2.ipynb

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


class OneHot(nn.Module):
    def __init__(self, MU):
        super(OneHot, self).__init__()
        self.MU = MU
        self.ones = torch.sparse.torch.eye(MU).to(device)

    def forward(self, x):
        x = x.to(device)
        x = x.long()
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
        ###Creating WAVENET blocks stacks###
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
        x.to(device)
        x = self.One_hot(x)
        ### shape of x [batch_size, n_category, seq_len]
        x = self.input_convs(x)  ### shape [batch_size, n_category, n_residual]
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

        if x.size(2) > self.seq_len_segment:
            x = x[:, :, :self.seq_len_segment]
        elif x.size(2) < self.seq_len_segment:
            padding = self.seq_len_segment - x.size(2)
            x = torch.nn.functional.pad(x, (0, padding))  # Pad

        return x

    def generate(self, input, num_samples=100):
        gen_list = input.data.tolist()
        for _ in tqdm(range(num_samples)):
            x = Variable(torch.LongTensor(gen_list[-sum(self.dilations) - 1:]))
            y = self.forward(x)
            _, i = y.max(dim=1)
            gen_list.append(i.data.tolist()[-1])
        return gen_list

    def generate_fix(self, input, num_samples=100):
        gen_list = input.data.tolist()
        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                x = torch.LongTensor(gen_list[-sum(self.dilations) - 1:])
                y = self.forward(x)
                _, i = y.max(dim=1)
                gen_list.append(i.tolist()[-1])
        return gen_list

class SnippetDatasetHSN(Dataset):
    def __init__(self, hsn, seq_len_segment, mu, scaling='minmax'):
        self.num_rows = 0
        self.size = seq_len_segment
        self.scaling = scaling
        self.seq_len_segment = seq_len_segment
        self.mu = mu
        self.data = self.createData(hsn)

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

        # same as above - ignore
        '''
        dataset = []
        for sample in data:
            sample = torch.from_numpy(sample)
            max_val = torch.max(sample)
            min_val = torch.min(sample)
            if max_val > torch.abs(min_val):
                sample = sample / max_val
            else:
                sample = sample / torch.abs(min_val)
            dataset.append(sample)
        '''
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
