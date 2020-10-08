import pytest
import os
import numpy as np
import pandas as pd
import torch

from stfs_pytoolbox.ML_Utils.loader.TabularLoader import TabularLoader


# Für die Test Cases extra kleine Datensätze erstellen, muss ein Loader Object inizialisieren, um manche Sachen zu testen ?
# keine Fehler aufnehmen, die durch verschiede Packages schon selbst erzeugt werden

class TestClass:
    def setup_method(self):
        example_df = pd.DataFrame(np.random.rand(4, 4))
        example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']
        self.example_df = example_df
        self.Loader = TabularLoader(example_df, features=['T_0', 'P_0'], labels=['yCO2', 'wH2O'])

    def test_init(self):
        with pytest.raises(AssertionError):
            Loader = TabularLoader(self.example_df, features=['T_0', 69], labels=['yCO2', 'wH2O'])
        with pytest.raises(AssertionError):
            Loader = TabularLoader(self.example_df, features=['T_0', 'P_0'], labels=['yCO2', 69])
        with pytest.raises(AssertionError):
            Loader = TabularLoader(self.example_df, features=['T_0', 'P_0'], labels=['T_0', 'wH2O'])
        with pytest.raises(KeyError):
            Loader = TabularLoader(self.example_df, features=['T_0', 'some other str'], labels=['yCO2', 'wH2O'])

    def test_add_val_data(self):
        Loader = self.Loader
        self.example_df.to_csv('example.csv')
        Loader.add_val_data('example.csv')
        os.remove('example.csv')

    def test_val_split(self):
        with pytest.raises(NameError):
            self.Loader.val_split(method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='explicit')
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='percentage')

    def test_add_test_data(self):
        Loader = self.Loader
        self.example_df.to_csv('example.csv')
        Loader.add_test_data('example.csv')
        os.remove('example.csv')

    def test_test_split(self):
        with pytest.raises(NameError):
            self.Loader.test_split(method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='explicit')
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='percentage')

    def test_train_dataloader(self):
        train_dataloader = self.Loader.train_dataloader()
        assert isinstance(train_dataloader, torch.utils.data.dataloader.DataLoader)

    def test_val_dataloader(self):
        self.Loader.val_split(method='random', val_size=0.25)
        val_dataloader = self.Loader.val_dataloader()
        assert isinstance(val_dataloader, torch.utils.data.dataloader.DataLoader)

    def test_test_dataloader(self):
        self.Loader.test_split(method='random', test_size=0.25)
        test_dataloader = self.Loader.test_dataloader()
        assert isinstance(test_dataloader, torch.utils.data.dataloader.DataLoader)

    def test_save_load(self):
        self.Loader.save('exampleLoader.pkg')
        Loader2 = TabularLoader.load('exampleLoader.pkg')
        assert isinstance(Loader2, TabularLoader)