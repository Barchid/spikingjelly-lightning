from typing import Optional
import pytorch_lightning as pl
from torch.functional import split
from torch.utils import data
from torch.utils.data import random_split, DataLoader
import tonic
from torchvision import transforms
import os
import numpy as np


class NMNISTDataModule(pl.LightningDataModule):
    def __init__(self, timesteps: int, batch_size: int, data_dir: str = "data/nmnist", **kwargs):
        super().__init__()
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.data_dir = data_dir

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

        # use a to_frame transform
        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.timesteps)
        self.transform = transforms.Compose(
            [frame_transform, transforms.Lambda(lambda x: (x > 0).astype(np.float32))])

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
        tonic.datasets.NMNIST(save_to=self.data_dir, train=True)
        tonic.datasets.NMNIST(save_to=self.data_dir, train=False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=True, transform=self.transform)
        self.val_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
