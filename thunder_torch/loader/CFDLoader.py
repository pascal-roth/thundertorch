#######################################################################################################################
# Loader for 2D (e.g. images)
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

from thunder_torch import _modules_models
from thunder_torch import _logger
from thunder_torch.loader import _utils
from thunder_torch.utils.option_class import OptionClass


class CFDLoader:
    """

    """

    def __init__(self, df_samples: pd.DataFrame, features: list, labels: list, **kwargs) -> None:
        """
       
        """
        self.lparams = Namespace()
        self.lparams.features = features
        self.lparams.labels = labels
        self.lparams.x_scaler = kwargs.pop('x_scaler', None)
        self.lparams.y_scaler = kwargs.pop('y_scaler', None)
        self.lparams.batch = kwargs.pop('batch', 64)
        self.lparams.num_workers = kwargs.pop('num_workers', 10)
        self.lparams.data_path = kwargs.pop('data_path', None)
        self.check_lparams()

        # self.samples = df_samples
        # self.x_train = self.samples[features]
        # self.y_train = self.samples[labels]
        self.x_train = df_samples[features]
        self.y_train = df_samples[labels]

        self.x_val = self.y_val = self.x_test = self.y_test = None
        if kwargs.get('val_split', None):
            self.val_split(**kwargs.pop('val_split'))
        elif kwargs.get('val_path', None):
            self.add_val_data(**kwargs.pop('val_path'))
        if kwargs.get('test_split', None):
            self.test_split(**kwargs.pop('test_split'))
        elif kwargs.get('test_path', None):
            self.add_test_data(**kwargs.pop('test_path'))

        if self.lparams.x_scaler is None:
            x_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.x_scaler = x_min_max_scaler.fit(self.x_train)
        if self.lparams.y_scaler is None:
            y_min_max_scaler = preprocessing.MinMaxScaler()
            self.lparams.y_scaler = y_min_max_scaler.fit(self.y_train)

        if len(kwargs) != 0:
            _logger.warning('Additional/ unexpected kwargs are given!')

    def check_lparams(self) -> None:
        assert all(isinstance(elem, str) for elem in self.lparams.features), "Given features is not a list of strings!"
        assert all(isinstance(elem, str) for elem in self.lparams.labels), "Given labels is not a list of strings!"
        assert all(elem not in self.lparams.labels for elem in self.lparams.features), "Feature is included in labels"

    # training_data ###################################################################################################
    def add_train_data(self, path) -> None:
        """
        Load training samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.data_path = path
        samples_train = _utils.read_df_from_file(path)
        if self.x_train is not None: _logger.warning('Train data overwritten')
        self.x_train = samples_train[self.lparams.features]
        self.y_train = samples_train[self.lparams.labels]
        _logger.debug(f'Train samples added from file {path}!')

    # validation_data #################################################################################################
    def add_val_data(self, path) -> None:
        """
        Load validation samples and separate them into input and target samples

        Parameters
        ----------
        val_samples     - file path
        """
        self.lparams.val = {'path': path}
        samples_val = _utils.read_df_from_file(path)
        if self.x_val is not None: _logger.warning('Validation data overwritten')
        self.x_val = samples_val[self.lparams.features]
        self.y_val = samples_val[self.lparams.labels]
        _logger.debug(f'Validation samples added from file {path}!')

    def val_split(self, **kwargs) -> None:
        """
        Split available samples into training and validation set

        Parameters
        ----------
        kwargs
            method: str                 - method to split training data
            params: float/ dict         - parameters of the split
        """
        self.lparams.val = {'method': kwargs.pop('method', 'random'), 'params': kwargs.pop('params', 0.2)}

        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val['method']) \
            (self.x_train, self.y_train, self.lparams.val['params'])
        _logger.debug('Validation set split performed!')

        if len(kwargs) != 0:
            _logger.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "params"')

    # test_data #######################################################################################################
    def add_test_data(self, path) -> None:
        """
        Load test samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.test = {'path': path}
        samples_test = _utils.read_df_from_file(path)
        if self.x_test is not None: _logger.warning('Test data overwritten')
        self.x_test = samples_test[self.lparams.features]
        self.y_test = samples_test[self.lparams.labels]
        _logger.debug(f'Test samples added from file {path}!')

    def test_split(self, **kwargs) -> None:
        """
        Split available samples into training and test set

        Parameters
        ----------
        kwargs
            method: str                 - method to split training data
            params: float/ dict         - parameters of the split
        """
        self.lparams.test = {'method': kwargs.pop('method', 'random'), 'params': kwargs.pop('params', 0.2)}

        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils,
                                                                       'data_split_' + self.lparams.test['method']) \
            (self.x_train, self.y_train, self.lparams.test['params'])
        _logger.debug('Test set split performed!')

        if len(kwargs) != 0:
            _logger.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "params"')

    # create pytorch dataloaders ######################################################################################
    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the training data (all kwargs of the PyTorch DataLoader can be used)
        """
        self.x_train = self.lparams.x_scaler.transform(self.x_train)
        self.y_train = self.lparams.y_scaler.transform(self.y_train)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_train), torch.tensor(self.y_train))
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the validation data (all kwargs of the PyTorch DataLoader can be used)
        """
        self.x_val = self.lparams.x_scaler.transform(self.x_val)
        self.y_val = self.lparams.y_scaler.transform(self.y_val)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_val), torch.tensor(self.y_val))
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Generate PyTorch DataLoader for the test data (all kwargs of the PyTorch DataLoader can be used)
        """
        self.x_test = self.lparams.x_scaler.transform(self.x_test)
        self.y_test = self.lparams.y_scaler.transform(self.y_test)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_test), torch.tensor(self.y_test))
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    # save and load TabluarLoader object ##############################################################################
    def save(self, filename) -> None:
        """
        Save CFDLoader cls as .pkl file

        Parameters
        ----------
        filename        - path path of .pkl file
        """
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.lparams.filename = filename
        _logger.info('CFDLoader object saved')

    @classmethod
    def load(cls, filename: str) -> object:
        with open(filename, 'rb') as input:
            return pickle.load(input)

    # classmethods ####################################################################################################
    @classmethod
    def read_from_file(cls, file, features: list, labels: list, **kwargs) -> object:
        """
        Create CFDLoader object from file

        Parameters
        ----------
        file            - file path
        features        - list of features
        labels          - list of labels
        kwargs          - see kwargs __init__

        Returns
        -------
        object          - CFDLoader object
        """
        df_samples = _utils.read_df_from_file(file)
        return cls(df_samples, features, labels, data_path=file,
                   **kwargs)

    @classmethod
    def read_from_yaml(cls, argsLoader: dict, **kwargs) -> object:
        """
        Create CFDLoader object from a dict similar to the one given under yml_template

        Parameters
        ----------
        argsLoader      - arguments to create the Loader
        kwargs          - see kwargs __init__

        Returns
        -------
        object          - CFDLoader object
        """
        options = CFDLoader.__get_OptionClass()
        OptionClass.checker(input_dict=argsLoader, option_classes=options)

        if 'load_DataLoader' in argsLoader:
            _, file_extention = os.path.splitext(argsLoader['load_DataLoader']['path'])
            if file_extention == '.pkl':
                Loader = CFDLoader.load(argsLoader['load_DataLoader']['path'])
                Loader.lparams.data_path = argsLoader['load_DataLoader']['path']
            elif file_extention == '.ckpt':
                Loader = CFDLoader.read_from_checkpoint(argsLoader['load_DataLoader']['path'])
            else:
                raise TypeError('Not supported file type to load DataLoader! Only supported are ".pkl" and ".ckpt"')

            if kwargs.get('batch'):
                Loader.lparams.batch = kwargs.pop('batch')
                _logger.info('Batch size stored in file in overwritten by kwargs argument')
            if kwargs.get('num_workers'):
                Loader.lparams.num_workers = kwargs.pop('num_workers')
                _logger.info('Num_workers stored in file in overwritten by kwargs argument')

        elif 'create_DataLoader' in argsLoader:
            argsCreate = argsLoader['create_DataLoader']

            # create Loader
            Loader = CFDLoader.read_from_file(argsCreate.pop('raw_data_path'), features=argsCreate.pop('features'),
                                                  labels=argsCreate.pop('labels'), **kwargs)

            # validation data
            if 'load_data' in argsCreate['validation_data']:
                Loader.add_val_data(**argsCreate['validation_data']['load_data'])
            elif 'split_data' in argsCreate['validation_data']:
                Loader.val_split(**argsCreate['validation_data']['split_data'])
            else:
                raise KeyError('No validation data selected! Either include dict "load_data" or "split_data".')

            # test data
            if 'load_data' in argsCreate['test_data']:
                Loader.add_test_data(**argsCreate['test_data']['load_data'])
            elif 'split_data' in argsCreate['test_data']:
                Loader.test_split(**argsCreate['test_data']['split_data'])
            else:
                raise KeyError('No test data selected! Either include dict "load_data" or "split_data".')

            # save loader
            if 'save_Loader' in argsCreate:
                Loader.save(argsCreate['save_Loader']['path'])

        else:
            raise KeyError('No DataLoader generated! Either include dict "load_DataLoader" or "create_DataLoader"!')

        return Loader

    @classmethod
    def read_from_checkpoint(cls, ckpt_file: str, model: str = 'LightningFlexMLP') -> object:
        """
        Create cls TabluarLoader from pytorch lightning checkpoint
        !! Hparams of the checkpoint had to be updated with lparams of the Loader in order to reconstruct the Loader!!

        Parameters
        ----------
        ckpt_file       - lightning checkpoint file

        Returns
        -------
        object          - CFDLoader object
        """
        model_cls = None
        for m in _modules_models:
            try:
                model_cls = getattr(importlib.import_module(m), model)
                _logger.debug(f'Model Class of type {model} has been loaded from {m}')
                break
            except AttributeError or ModuleNotFoundError:
                _logger.debug(f'Model Class of type {model} has NOT been loaded from {m}')

        assert model_cls is not None, f'Model {model} not found in {_modules_models}'

        pl_model = model_cls.load_from_checkpoint(ckpt_file)
        lparams = pl_model.hparams.lparams

        assert hasattr(lparams, 'data_path'), 'Data cannot be reloaded because the pass is missing'
        _, file_extention = os.path.splitext(lparams.data_path)

        if file_extention == '.pkl':
            Loader = CFDLoader.load(lparams.data_path)
        else:
            assert all(hasattr(lparams, elem) for elem in ['features', 'labels', 'batch', 'num_workers',
                                                           'x_scaler', 'y_scaler']), 'Parameters missing!'
            Loader = CFDLoader.read_from_file(lparams.data_path, features=lparams.features,
                                                  labels=lparams.labels, batch=lparams.batch,
                                                  num_workers=lparams.num_workers,
                                                  x_scaler=lparams.x_scaler, y_scaler=lparams.y_scaler)

        if 'path' in lparams.val:
            Loader.add_val_data(lparams.val.path)
        elif all(elem in lparams.val for elem in ['method', 'params']):
            Loader.val_split(method=lparams.val['method'], params=lparams.val['params'])
        else:
            raise KeyError('Keys to assign validation data are missing/ not complete')

        if 'path' in lparams.test:
            Loader.add_val_data(lparams.test.path)
        elif all(elem in lparams.test for elem in ['method', 'params']):
            Loader.val_split(method=lparams.test['method'], params=lparams.test['params'])
        else:
            raise KeyError('Keys to assign validation data are missing/ not complete')

        return Loader

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'DataLoader': OptionClass(template=CFDLoader.yaml_template(['DataLoader']))}
        options['DataLoader'].add_key('type', dtype=str, required=True)
        options['DataLoader'].add_key('load_DataLoader', dtype=dict, mutually_exclusive=['create_DataLoader'])
        options['DataLoader'].add_key('create_DataLoader', dtype=dict, mutually_exclusive=['load_DataLoader'])

        options['load_DataLoader'] = OptionClass(
            template=CFDLoader.yaml_template(['DataLoader', 'load_DataLoader']))
        options['load_DataLoader'].add_key('path', dtype=str, required=True)

        options['create_DataLoader'] = OptionClass(
            template=CFDLoader.yaml_template(['DataLoader', 'create_DataLoader']))
        options['create_DataLoader'].add_key('raw_data_path', dtype=str, required=True)
        options['create_DataLoader'].add_key('features', dtype=list, required=True)
        options['create_DataLoader'].add_key('labels', dtype=list, required=True)
        options['create_DataLoader'].add_key('validation_data', dtype=dict)
        options['create_DataLoader'].add_key('test_data', dtype=dict)
        options['create_DataLoader'].add_key('save_Loader', dtype=dict)

        options['validation_data'] = OptionClass(
            template=CFDLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                  'validation_data']))
        options['validation_data'].add_key('load_data', dtype=dict, mutually_exclusive=['split_data'])
        options['validation_data'].add_key('split_data', dtype=dict, mutually_exclusive=['load_data'])
        options['test_data'] = options['validation_data']
        options['test_data'].template = CFDLoader.yaml_template(['DataLoader', 'create_DataLoader', 'test_data'])

        options['load_data'] = OptionClass(template=CFDLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                 'validation_data', 'load_data']))
        options['load_data'].add_key('path', dtype=str, required=True)

        options['split_data'] = OptionClass(template=CFDLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                  'validation_data', 'split_data']))
        options['split_data'].add_key('method', dtype=str, required=True)
        options['split_data'].add_key('params', dtype=[float, dict], required=True, param_dict=True)

        options['save_Loader'] = OptionClass(template=CFDLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                   'save_Loader']))
        options['save_Loader'].add_key('path', dtype=str, required=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template of a CFDLoader object
        """
        template = {'DataLoader': {'type': 'CFDLoader',
                                   '###INFO###': 'load_DataLoader and create_DataLoader mutually exclusive',
                                   'load_DataLoader': {'path': 'name.pkl or modelXXX.ckpt'},
                                   'create_DataLoader': {'raw_data_path': 'samples_name.csv, .txt, .h5, .ulf',
                                                         'features': ['feature_1', 'feature_2', '...'],
                                                         'labels': ['label_1', 'label_2', '...'],
                                                         'validation_data':
                                                             {'###INFO###': 'load_data and split_data mutually exclusive',
                                                              'load_data': {'path': 'samples_name.csv, .txt, .h5, .ulf',
                                                                            'sep': 'separator (default: ","'},
                                                              'split_data': {'method': 'random/ percentage/ explicit',
                                                                             'params': 'split_params'}},
                                                         'test_data':
                                                             {'###INFO###': 'load_data and split_data mutually exclusive',
                                                              'load_data': {'path': 'samples_name.csv, .txt, .h5, .ulf',
                                                                            'sep': 'separator (default: ","'},
                                                              'split_data': {'method': 'random/ percentage/ explicit',
                                                                             'params': 'split_params'}},
                                                         'save_Loader': {'path': 'name.pkl'}}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)