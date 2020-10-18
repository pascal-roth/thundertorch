#######################################################################################################################
# Template to create DataLoader classes
#######################################################################################################################

# import packages
import os
import torch
import pickle
import logging
import yaml
from sklearn import preprocessing
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.models.LightningFlexMLP import LightningFlexMLP
from stfs_pytoolbox.ML_Utils.loader import _utils
from stfs_pytoolbox.ML_Utils.utils.utils_option_class import OptionClass


class DataLoaderTemplate:
    """
    Template class to create DataLoaders which get a certain input data, do preprocessing and can finally create PyTorch
    DataLoader that are used as input of the Lightning Trainer
    """

    def __init__(self, param_1, param_2, **kwargs) -> None:
        """
        Create class

        Parameters
        ----------
        param_1             - Example parameter
        param_2             - Example parameter
        """
        self.lparams = Namespace()  # Namespace object to group all important parameters of the DataLoader in order to
        # export them in the model checkpoint (and thus allow DataLoader rebuilt only with the model)
        self.lparams.param_1 = param_1
        self.lparams.param_2 = param_2
        self.lparams.kwargs_param_1 = kwargs.pop('kwargs_param_1', None)
        self.lparams.kwargs_param_2 = kwargs.pop('kwargs_param_2', None)

        self.check_lparams()

        # Definition of these samples categories necessary to generate dataloaders
        self.x_train = self.y_train = None
        self.x_val = self.y_val = None
        self.x_test = self.y_test = None

        if kwargs.get('val_split', None):
            self.val_split(**kwargs.pop('val_split'))
        elif kwargs.get('val_path', None):
            self.add_val_data(**kwargs.pop('val_path'))
        if kwargs.get('test_split', None):
            self.test_split(**kwargs.pop('test_split'))
        elif kwargs.get('test_path', None):
            self.add_test_data(**kwargs.pop('test_path'))

    def check_lparams(self):
        """
        function to check given parameters (e. g. with assert statements)
        """
        # assert param_1 == 'some value', 'Error Message'

    # training_data ###################################################################################################
    def add_train_data(self, path) -> None:
        """
        Function to load training samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.data_path = path
        if self.x_train is not None: logging.warning('Train data overwritten')
        self.x_train = self.y_train = None

    # validation_data #################################################################################################
    def add_val_data(self, path) -> None:
        """
        Function to load validation samples

        Parameters
        ----------
        val_samples     - file path
        """
        self.lparams.val_path = path
        if self.x_val is not None: logging.warning('Validation data overwritten')
        self.x_val = self.y_val = None

    def val_split(self, **kwargs) -> None:
        """
        Split available samples into training and validation set
        """
        self.lparams.val_method = kwargs.pop('method', 'some default method')
        self.lparams.val_params = kwargs.pop('val_params', 'some default parameters')

        # to split the samples, the function found in _utils.py can be used or own function can be written
        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val_method) \
            (self.x_train, self.y_train, self.lparams.val_params)

    # test_data #######################################################################################################
    def add_test_data(self, path) -> None:
        """
        Function to load test samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.test_path = path
        if self.x_test is not None: logging.warning('Test data overwritten')
        self.x_test = None
        self.y_test = None

    def test_split(self, **kwargs) -> None:
        """
        Split available samples into training and test set
        """
        self.lparams.test_method = kwargs.pop('method', 'some default method')
        self.lparams.test_params = kwargs.pop('test_params', 'some default parameters')

        # to split the samples, the function found in _utils.py can be used or own function can be written
        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils, 'data_split_' + self.lparams.test_method) \
            (self.x_train, self.y_train, self.lparams.test_params)

    # create pytorch dataloaders ######################################################################################
    # possible pre-processing steps can be added directly in the dataloader functions
    def train_dataloader(self, **kwargs):
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float())
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def val_dataloader(self, **kwargs):
        assert self.x_val is not None, 'Validation data has to be assigned before val_dataloader is created'
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_val).float(), torch.tensor(self.y_val).float())
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def test_dataloader(self, **kwargs):
        assert self.x_test is not None, 'Test data has to be assigned before test_dataloader is created'
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float())
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    # save and load TabluarLoader object ##############################################################################
    def save(self, filename) -> None:
        """
        Function to save DataLoader cls as .pkl file
        """
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.lparams.filename = filename

    @classmethod
    def load(cls, filename):
        """
        Funtion to reconstruct saved DataLoader
        """
        with open(filename, 'rb') as input:
            return pickle.load(input)

    # classmethods ####################################################################################################
    @classmethod
    def read_from_file(cls, file, **kwargs):
        """
        Function to create DataLoader from some kind of file
        """
        return cls(param_1=None, param_2=None, **kwargs)

    @classmethod
    def read_from_yaml(cls, argsLoader, **kwargs) -> object:
        options = DataLoaderTemplate.get_OptionClass()
        OptionClass.checker(input_dict=argsLoader, option_classes=options)

        if 'load_DataLoader' in argsLoader:
            _, file_extention = os.path.splitext(argsLoader['load_DataLoader']['path'])
            if file_extention == '.pkl':
                Loader = DataLoaderTemplate.load(argsLoader['load_DataLoader']['path'])
                Loader.lparams.data_path = argsLoader['load_DataLoader']['path']
            elif file_extention == '.ckpt':
                Loader = DataLoaderTemplate.read_from_checkpoint(argsLoader['load_DataLoader']['path'])
            else:
                raise TypeError('Not supported file type to load DataLoader! Only supported are ".pkl" and ".ckpt"')

            if kwargs['batch'] is not None:
                Loader.lparams.batch = kwargs.pop('batch')
                logging.info('Batch size stored in file in overwritten by kwargs argument')
            if kwargs['num_workers'] is not None:
                Loader.lparams.num_workers = kwargs.pop('num_workers')
                logging.info('Num_workers stored in file in overwritten by kwargs argument')

        elif 'create_DataLoader' in argsLoader:
            argsCreate = argsLoader['create_DataLoader']

            # create Loader
            Loader = DataLoaderTemplate.read_from_file(argsCreate.pop('raw_data_path'), features=argsCreate.pop('features'),
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
    def read_from_checkpoint(cls, ckpt_file):
        """
        Construct DataLoader with information saved in Model Checkpointing
        """
        model = LightningFlexMLP.load_from_checkpoint(ckpt_file)

        assert hasattr(model.hparams, 'data_path'), 'Data cannot be reloaded because the pass is missing'
        _, file_extention = os.path.splitext(model.hparams.data_path)

        if file_extention == '.pkl':
            Loader = DataLoaderTemplate.load(model.hparams.data_path)
        else:
            Loader = DataLoaderTemplate.read_from_file(model.hparams.data_path)

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

    @staticmethod
    def yaml_template(key_list):
        """
        Yaml Template for the class to allow usage
        Parameters
        ----------
        key_list
        """
        template = {'DataLoader': {'type': 'DataLoaderTemplate',
                                   'load_DataLoader': {'path': 'name.pkl or modelXXX.ckpt'},
                                   'create_DataLoader': {'raw_data_path': 'samples_name.csv, .txt, .h5, .flut', # TODO: change extension of flut datatype
                                                         'further_param_1': 'some information',
                                                         'further_param_2': 'some_information',
                                                         'validation_data': {'load_data': {'path': 'samples_name.csv, .txt, .h5'},
                                                                             'split_data': {'method': 'method name (pre implemented '
                                                                                                      'are random/percentage/explicit)',
                                                                                            'val_params': 'split_params'}},
                                                         'test_data': {'load_data': {'path': 'samples_name.csv, .txt, .h5, .flut'},
                                                                       'split_data': {'method': 'method name (pre implemented '
                                                                                                'are random/percentage/explicit)',
                                                                                      'test_params': 'split_params'}},
                                                         'save_Loader': {'execute': 'bool', 'path': 'name.pkl'}}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
