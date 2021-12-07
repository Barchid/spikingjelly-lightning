from typing import Optional
import pytorch_lightning as pl
from torch.functional import split
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from spikingjelly.datasets.n_mnist import NMNIST
import os


class NMNISTDataModule(pl.LightningDataModule):
    def __init__(self, timesteps: int, batch_size: int, data_dir: str = "data/nmnist", **kwargs):
        super().__init__()
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.data_dir = data_dir

        # create the directory if not exist
        os.makedirs(data_dir, exists_ok=True)

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use spikingjelly, all the download process is handled, we just have to make an instanciation
        NMNIST(self.data_dir, train=True, data_type='frame', frames_number=self.timesteps, split_by="number")
        NMNIST(self.data_dir, train=False, data_type='frame', frames_number=self.timesteps, split_by="number")

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = NMNIST(self.data_dir, train=True, data_type='frame',
                                    frames_number=self.timesteps, split_by="number")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.val_set = NMNIST(self.data_dir, train=False, data_type='frame',
                                  frames_number=self.timesteps, split_by="number")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)
