#######################################################################################################################
# DataLoader classes
#######################################################################################################################

# import packages
import os
import torch
import pickle
import logging
from sklearn import preprocessing
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.models.LightningFlexMLP import LightningFlexMLP
from stfs_pytoolbox.ML_Utils.loader import _utils


class TabularLoader:
    """
    Class to create pyTorch DataLoaders for Tabular input data
    """
    def __init__(self, df_samples, features, labels, **kwargs) -> None:
        """
        Create class

        Parameters
        ----------
        df_samples      - pd.DataFrame of samples
        features        - list of str: including features
        labels          - list of str: including labels
        """
        self.lparams = Namespace()
        self.lparams.x_scaler = kwargs.pop('x_scaler', None)
        self.lparams.y_scaler = kwargs.pop('y_scaler', None)
        self.lparams.batch = kwargs.pop('batch', 64)
        self.lparams.num_workers = kwargs.pop('num_workers', 10)
        self.lparams.data_path = kwargs.pop('data_path', None)
        self.lparams.features = features
        self.lparams.labels = labels
        assert all(isinstance(elem, str) for elem in self.lparams.features), "Given features is not a list of strings!"  # TODO: besser eine function zum checken der parameter derfinieren
        assert all(isinstance(elem, str) for elem in self.lparams.labels), "Given labels is not a list of strings!"
        assert all(elem not in self.lparams.labels for elem in self.lparams.features), "Feature is also included in labels"

        self.x_train = df_samples[features]
        self.y_train = df_samples[labels]

        self.x_val = self.y_val = self.x_test = self.y_test = None
        if kwargs.pop('val_split', None):  # TODO: schauen ob diese Method so funktioniert
            self.val_split(kwargs.pop('val_split'))
        if kwargs.pop('test_split', None):
            self.test_split(kwargs.pop('test_split'))

        if self.lparams.x_scaler is None:
            x_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.x_scaler = x_min_max_scaler.fit(self.x_train)
        if self.lparams.y_scaler is None:
            y_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.y_scaler = y_min_max_scaler.fit(self.y_train)

    # validation_data #################################################################################################
    def add_val_data(self, path) -> None:
        """
        Load validation samples and separate them into input and target samples

        Parameters
        ----------
        val_samples     - file path
        """
        self.lparams.val_path = path
        samples_val = _utils.read_df_from_file(path)
        if self.x_val is not None: logging.warning('Validation data overwritten')
        self.x_val = samples_val[self.lparams.features]
        self.y_val = samples_val[self.lparams.labels]

    def val_split(self, **kwargs):
        """
        Split available samples into training and validation set
        """
        self.lparams.val_method = kwargs.pop('method', 'random')
        self.lparams.val_params = kwargs.pop('params', 0.2)

        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val_method)\
            (self.x_train, self.y_train, self.lparams.val_params)

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "val_params"')

    # test_data #######################################################################################################
    def add_test_data(self, path):
        """
        Load test samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.test_path = path
        samples_test = _utils.read_df_from_file(path)
        self.x_test = samples_test[self.lparams.features]
        self.y_test = samples_test[self.lparams.labels]

    def test_split(self, **kwargs):
        """
        Split available samples into training and test set
        """
        self.lparams.test_method = kwargs.pop('method', 'random')
        self.lparams.test_params = kwargs.pop('params', 0.2)

        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils, 'data_split_' + self.lparams.test_method)\
            (self.x_train, self.y_train, self.lparams.test_params)

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "test_params"')

    # create pytorch dataloaders ######################################################################################
    def train_dataloader(self, **kwargs):  # TODO: memory consumption wenn original samples drin bleiben
        if self.x_val is None: self.val_split()  # TODO: maybe find a better solution to add an default
        if self.x_test is None: self.test_split()

        self.x_train = self.lparams.x_scaler.transform(self.x_train)
        self.y_train = self.lparams.y_scaler.transform(self.y_train)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float())  # TODO: find solution, Julians wants it without float() but then error
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def val_dataloader(self, **kwargs):
        self.x_val = self.lparams.x_scaler.transform(self.x_val)
        self.y_val = self.lparams.y_scaler.transform(self.y_val)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_val).float(), torch.tensor(self.y_val).float())
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def test_dataloader(self, **kwargs):
        self.x_test = self.lparams.x_scaler.transform(self.x_test)
        self.y_test = self.lparams.y_scaler.transform(self.y_test)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float())
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    # save and load TabluarLoader object ##############################################################################
    def save(self, filename) -> None:
        """
        Save TabularLoader cls as .pkl file

        Parameters
        ----------
        filename        - path path of .pkl file
        """
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.lparams.filename = filename

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as input:
            return pickle.load(input)

    # classmethods ####################################################################################################
    @classmethod
    def read_from_file(cls, file, features, labels, **kwargs):  # TODO: Einzelheiten über Dataformat heraufinden !
        df_samples = _utils.read_df_from_file(file)  # TODO: check delimiter (online function finden)
        return cls(df_samples, features, labels, data_path=file, **kwargs)  # TODO: nachschauen, ob Weitergabe der kwargs funktioniert

    @classmethod
    def read_from_yaml(cls, argsLoader, **kwargs):
        execute_flags = (argsLoader.load_DataLoader['execute'], argsLoader.create_DataLoader['execute'])
        assert len([execute_flags.index(True)]) == 1, 'Not valid! Chose if loader either loaded or created!'

        if argsLoader.load_DataLoader['execute']:
            _, file_extention = os.path.splitext(argsLoader.load_DataLoader['path'])
            if file_extention == '.pkg':
                Loader = TabularLoader.load(argsLoader.load_DataLoader['path'])
                Loader.lparams.data_path = argsLoader.load_DataLoader['path']
            elif file_extention == '.ckpg':
                Loader = TabularLoader.read_from_checkpoint(argsLoader.load_DataLoader['path'])
            else:
                raise TypeError('Not supported file type to load DataLoader! Only supported are ".pkg" and ".ckpg"')

        elif argsLoader.create_DataLoader['execute']:
            argsLoader = Namespace(**argsLoader.create_DataLoader)

            Loader = TabularLoader.read_from_file(argsLoader.raw_data_path, features=argsLoader.features,
                                                  labels=argsLoader.labels, **kwargs)

            if argsLoader.validation_data['load_data'].pop('execute'):
                Loader.add_val_data(**argsLoader.validation_data['load_data'])
            elif argsLoader.validation_data['split_data'].pop('execute'):
                Loader.val_split(**argsLoader.validation_data['split_data'])
            else:
                raise SyntaxError('No validation data selected! Either set execute flag in load or split data to "True"!')

            if argsLoader.test_data['load_data'].pop('execute'):
                Loader.add_test_data(**argsLoader.test_data['load_data'])
            elif argsLoader.test_data['split_data'].pop('execute'):
                Loader.test_split(**argsLoader.test_data['split_data'])
            else:
                raise SyntaxError('No test data selected! Either set execute flag in load or split data to "True".')

            if argsLoader.save_Loader['execute']:
                Loader.save(argsLoader.save_Loader['path'])

        return Loader

    @classmethod
    def read_from_checkpoint(cls, ckpt_file):
        """
        Create cls TabluarLoader from pytorch lightning checkpoint
        !! Hparams of the checkpoint had to be updated with lparams of the Loader in order to reconstruct the Loader!!

        Parameters
        ----------
        ckpt_file       - lightning checkpoint file

        Returns
        -------
        Loader          - TabularLoader class object
        """
        model = LightningFlexMLP.load_from_checkpoint(ckpt_file)

        _, file_extention = os.path.splitext(model.hparams.data_path)

        if file_extention == '.pkl':
            Loader = TabularLoader.load(model.hparams.data_path)
        else:
            Loader = TabularLoader.read_from_file(model.hparams.data_path, features=model.hparams.features,
                                                  labels=model.hparams.labels, batch=model.hparams.batch,
                                                  num_workers=model.hparams.num_workers,
                                                  x_scaler=model.hparams.x_scaler, y_scaler=model.hparams.y_scaler)

        if hasattr(model.hparams, 'val_path'):
            Loader.add_val_data(model.hparams.val_path)
        elif all(hasattr(model.hparams, attr) for attr in ['val_method', 'val_params']):
            Loader.val_split(method=model.hparams.val_method, val_params=model.hparams.val_params)
        else:
            raise KeyError('Keys to assign validation data are missing/ not complete')

        if hasattr(model.hparams, 'test_path'):
            Loader.add_val_data(model.hparams.test_path)
        elif all(hasattr(model.hparams, attr) for attr in ['test_method', 'test_params']):
            Loader.val_split(method=model.hparams.test_method, val_params=model.hparams.test_params)
        else:
            raise KeyError('Keys to assign validation data are missing/ not complete')

        return Loader

