import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import lightning as pl

class AEA2(pl.LightningModule):
    def __init__(self, input_size=(128, 20), lr = 3e-4,store_path = ""):
        super().__init__()
        self.dim = 64
        self.store_path = store_path
        self.lr = lr
        self.input_size = input_size
        self.num_fet = input_size[0]
        self.seq_len = input_size[1]
        self.encoder = nn.Sequential(
            nn.Conv1d(self.num_fet, self.dim, 3, stride=2),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),
        )

        self.flat_size, self.encoder_output_shape = self._infer_flat_size()
        print(self.flat_size, self.encoder_output_shape)

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, self.dim//8),
            nn.BatchNorm1d(self.dim//8),
            nn.LeakyReLU(),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.dim//8, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, self.dim, 3, stride=2, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(self.dim, 1, 3, stride=2, padding=0),
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

    def encode(self,x):
        x = self.encoder(x)
        x = self.encoder_fc(x.view(-1, self.flat_size))
        return x
    
    def decode(self,x):
        x = self.decoder_fc(x)
        x = self.decoder(x.view(-1, *self.encoder_output_shape))
        return x


    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def training_step(self, batch):
       
        input = batch.unsqueeze(1).float()
        pred = self.forward(input).squeeze()
        recon_loss = torch.sqrt(F.mse_loss(pred, batch[:,1:]))
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
    
    def _infer_flat_size(self):
        print(torch.ones(1, *self.input_size).shape)
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        print(encoder_output.shape)
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]