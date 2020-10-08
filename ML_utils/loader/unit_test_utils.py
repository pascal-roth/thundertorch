import pytest
import os

from stfs_pytoolbox.ML_Utils.loader._utils import *


class TestClass:
    def setup_method(self):
        self.example_df = pd.DataFrame(np.random.rand(4, 4))
        self.example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']
        self.x_samples = self.example_df[['T_0', 'P_0']]
        self.y_samples = self.example_df[['yCO2', 'wH2O']]

    def test_read_df_from_file(self):
        # test format .csv
        self.example_df.to_csv('example.csv')
        df_samples = read_df_from_file('example.csv')
        assert isinstance(df_samples, pd.DataFrame)
        os.remove('example.csv')

        # test format .h5
        self.example_df.to_hdf('example.h5', key='example')
        df_samples = read_df_from_file('example.h5')
        assert isinstance(df_samples, pd.DataFrame)
        os.remove('example.h5')

        # test format .txt
        self.example_df.to_csv('example.txt')
        df_samples = read_df_from_file('example.txt')
        assert isinstance(df_samples, pd.DataFrame)
        os.remove('example.txt')

        # test format .flut
        #  TODO: implement method if flut datatype is know

        with pytest.raises(TypeError):
            df_samples = read_df_from_file('some_file.other_type')

    def test_split_data_random(self):
        with pytest.raises(AssertionError):
            data_split_random(x_samples=self.x_samples, y_samples=self.y_samples, split_params=1.2)
        with pytest.raises(AssertionError):
            data_split_random(x_samples=self.x_samples, y_samples=self.y_samples, split_params='str')
    def test_split_data_percentage(self):
        with pytest.raises(AssertionError):
            data_split_percentage(x_samples=self.x_samples, y_samples=self.y_samples, split_params={'T_0': 0.05})
        with pytest.raises(AssertionError):
            data_split_percentage(x_samples=self.x_samples, y_samples=self.y_samples, split_params={'T_0': 1.2})

    def test_split_data_explicit(self):
        with pytest.raises(TypeError):
            data_split_explicit(x_samples=self.x_samples, y_samples=self.y_samples, split_params={'T_0': 'false type'})
        with pytest.raises(AssertionError):
            data_split_explicit(x_samples=self.x_samples, y_samples=self.y_samples, split_params={'T_0': 5})
