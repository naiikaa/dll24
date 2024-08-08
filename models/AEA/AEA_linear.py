import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels,int(channels//4), batch_first=True)
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

class AEA_linear(pl.LightningModule):
    def __init__(self, size, lr = 3e-4,store_path = ""):
        super().__init__()

        self.store_path = store_path
        self.lr = lr
        
        self.encoder = nn.Sequential(
            nn.Linear(size,size//4),
            nn.LeakyReLU(),
            nn.Linear(size//4,size//8),
            nn.LeakyReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(size//8,size//8),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(size//8,size//4),
            nn.LeakyReLU(),
            nn.Linear(size//4,size),
            nn.LeakyReLU(),
        )
            


        #self.net = nn.Sequential(self.encoder,self.attention,self.decoder)

    def saveModel(self, model, path):
        """Saves state of an model for future loading

        Args:
            model (Neural Network): The model that is saved
            path (String): Location where to save it
        """
        torch.save(model.state_dict(), path)

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        #x_hat = self.fitter(x_hat.squeeze())
        return x
    
    def encode(self,x):
        x = self.encoder(x)
        return self.attention(x)
    
    def decode(self,x):
       return self.decoder(x)

    def training_step(self, batch):
       
        input = batch.unsqueeze(1).float()
        pred = self.forward(input).squeeze()
      
        recon_loss = torch.sqrt(F.mse_loss(pred, batch))
        loss = recon_loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer