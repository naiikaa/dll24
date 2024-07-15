
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class TaskEmbedding(nn.Module):
    """Embed a single categorical predictor
    
    Keyword Arguments:
    
    num_output_classes: int, number of output classes
    num_cat_classes: list[int], number of classes for each categorical variable
    num_cont: int, number of continuous variables
    embedding_dim: int, dimension of the embedding
    hidden_dim: int, dimension of the hidden layer
    """
    def __init__(self, num_output_classes:int, num_cat_classes:list[int], num_cont:int, embedding_dim:int=64, hidden_dim:int=64):
        super().__init__()
        # Create an embedding for each categorical input
        self.embeddings = nn.ModuleList([nn.Embedding(nc, embedding_dim) for nc in num_cat_classes])
        self.fc1 = nn.Linear(in_features=len(num_cat_classes) * embedding_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=num_cont, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(2*hidden_dim, num_output_classes)
        
    def forward(self, x_cat, x_con):
        # Embed each of the categorical variables

        x_embed = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, dim=1)
        x = self.fc1(x_embed)
        x_con = self.fc2(x_con)
        x = torch.cat([x_con, x], dim=1)
        x = self.relu(x)
        return self.out(x)    

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size )
        x = x.swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
       
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class CAUnet(nn.Module):
    def __init__(self,n_steps,c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.n_steps = n_steps
        self.time_dim = time_dim
        
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)
        self.bot1 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 256)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


class UNet_conditional(CAUnet):
    def __init__(self,n_steps, c_in=3, c_out=3, time_dim=256, num_classes=None,num_cont=None, **kwargs):
        super().__init__(n_steps,c_in, c_out, time_dim, **kwargs)
        self.num_classes = num_classes
        
        if num_classes is not None:
            self.task_emb  = TaskEmbedding(time_dim,num_classes,num_cont)
            
    def forward(self, x, t, y,cont):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        if y is not None:
            t += self.task_emb(y,cont)
        
        return self.unet_forward(x, t)