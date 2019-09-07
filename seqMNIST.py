# Based on an example on the PyTorch Lightning homepage:
# https://github.com/williamFalcon/pytorch-lightning#how-do-i-do-use-it

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl


class SeqMNIST(pl.LightningModule):

    def __init__(self, model, learning_rate, default_batch_size):
        super(SeqMNIST, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.default_batch_size = default_batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
                          batch_size=self.default_batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
                          batch_size=self.default_batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
                          batch_size=self.default_batch_size)
