import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class PandasDataModule(pl.LightningDataModule):

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, features = None, labels = None, batch = 32):
        """

        Parameters
        ----------
        train_data: pd.DataFrame
            dataframe that contains training data
        val_data: pd.DataFrame
            DataFrame that contains validation data, if not provided train_data is used
        features: list of strings
            list of feature names
        labels: list of strings
            list of label names
        batch: int default=32
            batch size for training
        """
        super().__init__()

        if not features:

        if not labels:



    def prepare_data(self):
        # possible download something but nothing to do here
        # is run once (!) on GPU1
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)import pytorch_lightning as pl
        # from torch.utils.data import random_split, DataLoader
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)