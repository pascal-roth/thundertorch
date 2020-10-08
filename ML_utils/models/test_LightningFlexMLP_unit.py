import pytest
import os
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.loader import TabularLoader
from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP


# Für die Test Cases extra kleine Datensätze erstellen, muss ein Loader Object inizialisieren, um manche Sachen zu testen ?
# keine Fehler aufnehmen, die durch verschiede Packages schon selbst erzeugt werden

class TestClass:
    def setup_method(self):
        self.example_df = pd.DataFrame(np.random.rand(6, 4))
        self.example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']

    def get_loader(self):
        Loader = TabularLoader(self.example_df)
        Loader.val_split(method='sample', split_method='percentage', val_params={'T_0': 0.20})
        Loader.test_split(method='sample', split_method='percentage', test_params={'T_0': 0.20})
        return Loader

    def get_hparams(self):
        hparams = Namespace()
        hparams.features = ['T_0', 'P_0']
        hparams.labels = ['yCO2', 'wH2O']
        hparams.hidden_layer = [16, 16]
        hparams.batch = 16
        hparams.lr = 1.e-3
        hparams.output_relu = False
        hparams.activation = 'relu'
        hparams.loss = 'MSE'
        hparams.optimizer = 'adam'
        hparams.scheduler = False
        hparams.num_workers = 4
        hparams.x_scaler = None
        hparams.y_scaler = None
        return hparams

    def test_prepare_data(self):
        hparams = self.get_hparams()
        with pytest.raises(KeyError):
            Loader = self.get_loader()
            Loader.samples_train = None
            model01 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer01 = pl.Trainer(fast_dev_run=True)
            trainer01.fit(model01)
        with pytest.raises(KeyError):
            Loader = self.get_loader()
            Loader.samples_val = None
            model02 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer02 = pl.Trainer(fast_dev_run=True)
            trainer02.fit(model02)
        with pytest.raises(KeyError):
            Loader = self.get_loader()
            Loader.samples_test = None
            model03 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer03 = pl.Trainer(fast_dev_run=True)
            trainer03.fit(model03)
        os.remove('lightning_logs')

    def test_activation_fn(self):
        with pytest.raises(NotImplementedError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.activation = 'some other str'
            model01 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer01 = pl.Trainer(fast_dev_run=True)
            trainer01.fit(model01)
        with pytest.raises(AssertionError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.activation = 99
            model02 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer02 = pl.Trainer(fast_dev_run=True)
            trainer02.fit(model02)
        os.remove('lightning_logs')

    def test_loss_fn(self):
        with pytest.raises(NotImplementedError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.loss = 'some other str'
            model01 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer01 = pl.Trainer(fast_dev_run=True)
            trainer01.fit(model01)
        with pytest.raises(AssertionError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.loss = 99
            model02 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer02 = pl.Trainer(fast_dev_run=True)
            trainer02.fit(model02)
        os.remove('lightning_logs')

    def test_configure_optimizer(self):
        with pytest.raises(NotImplementedError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.optimizer = 'some other str'
            model01 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer01 = pl.Trainer(fast_dev_run=True)
            trainer01.fit(model01)
        with pytest.raises(AssertionError):
            Loader = self.get_loader()
            hparams = self.get_hparams()
            hparams.optimizer = 99
            model02 = LightningFlexMLP(hparams=hparams, TabularLoader=Loader)
            trainer02 = pl.Trainer(fast_dev_run=True)
            trainer02.fit(model02)
        os.remove('lightning_logs')
