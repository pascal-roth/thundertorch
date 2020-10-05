import pytest
import os
import numpy as np
import pandas as pd

from stfs_pytoolbox.ML_Utils.TabularLoader import *


# Für die Test Cases extra kleine Datensätze erstellen, muss ein Loader Object inizialisieren, um manche Sachen zu testen ?
# keine Fehler aufnehmen, die durch verschiede Packages schon selbst erzeugt werden

class TestClass:
    def setup_method(self):
        example_df = pd.DataFrame(np.random.rand(4, 4))
        example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']
        self.example_df = example_df
        self.Loader = TabularLoader(example_df)

    def test_add_val_data(self):
        with pytest.raises(AssertionError):
            self.Loader.add_val_data(np.zeros((3, 3)))

    def test_val_split(self):
        with pytest.raises(NameError):
            self.Loader.val_split(method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='random', val_size=1.2)
        with pytest.raises(TypeError):
            self.Loader.val_split(method='random', val_size='str')
        with pytest.raises(NameError):
            self.Loader.val_split(method='sample', split_method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='sample', split_method='percentage', val_params={'T_0': 1.2})
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='sample', split_method='percentage', val_params={'T_0': 0.05})
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='sample', split_method='explicit', val_params={'T_0': 5})
        with pytest.raises(AssertionError):
            self.Loader.val_split(method='sample', split_method='explicit', val_params={'T_0': [1, 2, 3, 4, 5]})
        with pytest.raises(KeyError):
            self.Loader.val_split(method='sample', split_method='explicit', val_params={'T_0': 'false type'})


    def test_add_test_data(self):
        with pytest.raises(AssertionError):
            self.Loader.add_test_data(np.zeros((3, 3)))

    def test_test_split(self):
        with pytest.raises(NameError):
            self.Loader.test_split(method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='random', test_size=1.2)
        with pytest.raises(TypeError):
            self.Loader.test_split(method='random', test_size='str')
        with pytest.raises(NameError):
            self.Loader.test_split(method='sample', split_method='some other str')
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='sample', split_method='percentage', test_params={'T_0': 1.2})
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='sample', split_method='percentage', test_params={'T_0': 0.05})
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='sample', split_method='explicit', test_params={'T_0': 5})
        with pytest.raises(AssertionError):
            self.Loader.test_split(method='sample', split_method='explicit', test_params={'T_0': [1, 2, 3, 4, 5]})
        with pytest.raises(KeyError):
            self.Loader.val_split(method='sample', split_method='explicit', val_params={'T_0': 'false type'})

    def test_classmethod_csv(self):
        self.example_df.to_csv('example.csv')
        Loader = TabularLoader.read_from_csv('example.csv')
        os.remove('example.csv')

    def test_classmethod_h5(self):
        self.example_df.to_hdf('example.h5', key='example')
        Loader = TabularLoader.read_from_h5('example.h5')
        os.remove('example.h5')

    def test_classmethod_flut(self):
        pass  # TODO: implement test if data format is known
