import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels,int(channels/4), batch_first=True)
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

class AEA(pl.LightningModule):
    def __init__(self, num_fet,less_param = False, lr = 3e-4,store_path = ""):
        super().__init__()

        self.store_path = store_path
        self.less_param = less_param
        self.lr = lr
        if less_param:
            self.encoder = nn.Sequential(
                nn.Conv1d(num_fet, 32, 3, stride=2, padding=1),
                SelfAttention(32),
                nn.LeakyReLU(),
                nn.Conv1d(32, 64, 3, stride=2, padding=1),
                SelfAttention(64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, 3, stride=2, padding=1),
                nn.LeakyReLU(),
            )

            self.attention = SelfAttention(128)

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1),
                SelfAttention(64),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1),
                SelfAttention(32),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(32, 1, 3, stride=2, padding=1),
                nn.Sigmoid(),
            )
            
            self.fitter = nn.Linear(153,120)
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(num_fet, 32, 3, stride=1, padding=0),
                SelfAttention(32),
                nn.LeakyReLU(),
                nn.Conv1d(32, 64, 3, stride=1, padding=0),
                SelfAttention(64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, 3, stride=1, padding=0),
                SelfAttention(128),
                nn.LeakyReLU(),
                nn.Conv1d(128, 256, 3, stride=1, padding=0),
                nn.LeakyReLU(),
            )

            self.attention = SelfAttention(256)

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(256, 128, 3, stride=1, padding=0),
                SelfAttention(128),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(128, 64, 3, stride=1, padding=0),
                SelfAttention(64),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 32, 3, stride=1, padding=0),
                SelfAttention(32),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(32, 1, 3, stride=1, padding=0),
                nn.Sigmoid(),
            )
            
            #self.fitter = nn.Linear(24993,160000)

            self.net = nn.Sequential(self.encoder,self.attention,self.decoder)

    def saveModel(self, model, path):
        """Saves state of an model for future loading

        Args:
            model (Neural Network): The model that is saved
            path (String): Location where to save it
        """
        torch.save(model.state_dict(), path)

    def forward(self,x):

        #x_hat = self.fitter(x_hat.squeeze())
        return self.net(x)

    def training_step(self, batch):
        input = batch[0].permute(0,2,1).float()
        target = batch[2].float()
        pred = self.forward(input).squeeze()
        recon_loss = torch.sqrt(F.mse_loss(pred, target))
        loss = recon_loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        input = batch[0].permute(0,2,1).float()
        target = batch[2].float()
        pred = self.forward(input).squeeze()
        recon_loss = torch.sqrt(F.mse_loss(pred, target))
        loss = recon_loss
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch):
        input = batch[0].permute(0,2,1).float()
        target = batch[2].float()
        pred = self.forward(input).squeeze()
        recon_loss = torch.sqrt(F.mse_loss(pred, target))
        loss = recon_loss
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer