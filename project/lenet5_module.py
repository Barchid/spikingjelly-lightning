from argparse import ArgumentParser
from spikingjelly.datasets.n_mnist import NMNIST

import torch
import pytorch_lightning as pl
from torch.nn import functional as F

from project.models.spiking_lenet5 import SpikingLeNet5


class Module(pl.LightningModule):
    def __init__(self, learning_rate: float, neuron_model: str, bias: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = SpikingLeNet5(
            in_channels=2,
            num_classes=10,
            height=NMNIST.get_H_W()[0],
            width=NMNIST.get_H_W()[1],
            neuron_model=self.hparams.neuron_model,
            bias=self.hparams.bias
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--bias', action="store_true")
        parser.add_argument('--neuron_model', choices=["LIF", "PLIF", "IF"], default="LIF")
        return parser
