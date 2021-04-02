#######################################################################################################################
# DataLoaderBase
#######################################################################################################################

# import packages
import os
import torch
import pickle
import yaml
import importlib
import pandas as pd
from sklearn import preprocessing
from argparse import Namespace
from typing import Union, Tuple
from pathlib import Path

from abc import ABC, abstractmethod

from thunder_torch import _modules_models
from thunder_torch import _logger
from thunder_torch import models
from thunder_torch.loader import _utils
from thunder_torch.utils.option_class import OptionClass


class DataLoaderBase(ABC):
    """
    DataLoader Base of the Toolbox, includes repeating functions and functionalities

    Supported possibilities to create a TabularLoader object
    ----------------------------------------------
    - read from a file (possible dtypes: ".csv", ".txt", ".h5")
    - load a saved DataLoader
    - restore from model checkpoint (NOTE: in order to restore a Loader using a ckpt, the lparams have to be saved
    in the hparams namespace of the model. Thereby, the hparams update fkt can be used. An example is found under
    thunder_torch.utils.utils_execute.get_dataLoader)
    """

    def __init__(self) -> None:

        self.lparams = Namespace()
        self.lparams.x_scaler = None
        self.lparams.y_scaler = None
        self.lparams.batch = None
        self.lparams.num_workers = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def get_scaler(self):
        if self.lparams.x_scaler is None:
            x_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.x_scaler = x_min_max_scaler.fit(self.x_train)
        if self.lparams.y_scaler is None:
            y_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.y_scaler = y_min_max_scaler.fit(self.y_train)

    # training_data ###################################################################################################
    @abstractmethod
    def add_train_data(self, *args, **kwargs) -> None:
        """
        Load training samples and separate them into input and target samples
        """
        pass

    # validation_data #################################################################################################
    @abstractmethod
    def add_val_data(self, *args, **kwargs) -> None:
        """
        Load validation samples and separate them into input and target samples
        """
        pass

    def val_split(self, method: str = 'random', params: Union[dict, float] = 0.2) -> None:
        """
        Split available samples into training and validation set

        Parameters
        ----------
        method:             - method to split training data
        params:             - parameters of the split
        """
        self.lparams.val = {'method': method, 'params': params}

        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val['method']) \
            (self.x_train, self.y_train, self.lparams.val['params'])
        _logger.debug('Validation set split performed!')

    # test_data #######################################################################################################
    @abstractmethod
    def add_test_data(self, *args, **kwargs) -> None:
        """
        Load test samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        sep              - separator
        """
        pass

    def test_split(self, method: str = 'random', params: Union[dict, float] = 0.2) -> None:
        """
        Split available samples into training and test set

        Parameters
        ----------
        method:           - method to split training data
        params:           - parameters of the split
        """
        self.lparams.test = {'method': method, 'params': params}

        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils, 'data_split_' + self.lparams.test['method']) \
            (self.x_train, self.y_train, self.lparams.test['params'])
        _logger.debug('Test set split performed!')

    # create pytorch dataloaders ######################################################################################
    def data_normalization(self, x_samples: pd.DataFrame, y_samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.lparams.x_scaler and not self.lparams.y_scaler: self.get_scaler()
        x_samples = self.lparams.x_scaler.transform(x_samples)
        y_samples = self.lparams.y_scaler.transform(y_samples)
        return x_samples, y_samples

    @staticmethod
    def get_tensorDataset(x_samples: pd.DataFrame, y_samples: pd.DataFrame) -> torch.utils.data.TensorDataset:
        return torch.utils.data.TensorDataset(torch.tensor(x_samples), torch.tensor(y_samples))

    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the training data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_train, self.y_train)
        tensor = self.get_tensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the validation data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_val, self.y_val)
        tensor = self.get_tensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the test data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_test, self.y_test)
        tensor = self.get_tensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    # save and load TabluarLoader object ##############################################################################
    def save(self, filename: Union[str, Path]) -> None:
        """
        Save TabularLoader cls as .pkl file

        Parameters
        ----------
        filename        - path path of .pkl file
        """
        if not isinstance(filename, str):
            filename = str(filename)
        if not filename.lower().endswith('.pkl'):
            filename = filename + '.pkl'

        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.lparams.filename = filename
        _logger.info('TabularLoader object saved')

    @classmethod
    def load(cls, filename: Union[str, Path]) -> object:
        if not isinstance(filename, str):
            filename = str(filename)
        if not filename.lower().endswith('.pkl'):
            filename = filename + '.pkl'

        with open(filename, 'rb') as input:
            return pickle.load(input)

    # classmethods ####################################################################################################
    @classmethod
    def read_from_file(cls, *args, **kwargs) -> object:
        """
        Create DataLoader object from file
        """
        pass

    @classmethod
    def read_from_yaml(cls, argsLoader: dict, **kwargs) -> object:
        """
        Create TabularLoader object from a dict similar to the one given under yml_template
        """
        pass

    @classmethod
    def read_from_checkpoint(cls, ckpt_file: str, model: str = 'LightningFlexMLP') -> object:
        """
        Create cls DataLoader from pytorch lightning checkpoint
        !! Hparams of the checkpoint had to be updated with lparams of the Loader in order to reconstruct the Loader!!

        Parameters
        ----------
        ckpt_file       - lightning checkpoint file

        Returns
        -------
        object          - TabularLoader object
        """
        pass

    @staticmethod
    def __get_OptionClass() -> dict:
        pass

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template of a TabularLoader object
        """
        pass