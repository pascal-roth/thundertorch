#######################################################################################################################
# DataLoader classes
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
        self.lparams = Namespace()  # TODO: Idea to allow initialization without data and make function to add training data
        self.lparams.x_scaler = kwargs.pop('x_scaler', None)
        self.lparams.y_scaler = kwargs.pop('y_scaler', None)
        self.lparams.batch = kwargs.pop('batch', 64)
        self.lparams.num_workers = kwargs.pop('num_workers', 10)
        self.lparams.data_path = kwargs.pop('data_path', None)
        self.lparams.features = features
        self.lparams.labels = labels
        self.check_lparams()

        # self.samples = df_samples  # TODO: mit Julian besprechen, weil führt zu einem increase in memory consumption
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

    def check_lparams(self):
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
        if self.x_train is not None: logging.warning('Train data overwritten')
        self.x_train = samples_train[self.lparams.features]
        self.y_train = samples_train[self.lparams.labels]

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

    def val_split(self, **kwargs) -> None:
        """
        Split available samples into training and validation set
        """
        self.lparams.val_method = kwargs.pop('method', 'random')
        self.lparams.val_params = kwargs.pop('val_params', 0.2)

        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val_method) \
            (self.x_train, self.y_train, self.lparams.val_params)

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "val_params"')

    # test_data #######################################################################################################
    def add_test_data(self, path) -> None:
        """
        Load test samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        """
        self.lparams.test_path = path
        samples_test = _utils.read_df_from_file(path)
        if self.x_test is not None: logging.warning('Test data overwritten')
        self.x_test = samples_test[self.lparams.features]
        self.y_test = samples_test[self.lparams.labels]

    def test_split(self, **kwargs) -> None:
        """
        Split available samples into training and test set
        """
        self.lparams.test_method = kwargs.pop('method', 'random')
        self.lparams.test_params = kwargs.pop('test_params', 0.2)

        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils, 'data_split_' + self.lparams.test_method) \
            (self.x_train, self.y_train, self.lparams.test_params)

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "test_params"')

    # create pytorch dataloaders ######################################################################################
    def train_dataloader(self, **kwargs):
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
        assert self.x_test is not None, 'Test data has to be assigned before test_dataloader is created'  # TODO: schauen ob dann default genommen werden kann, wenn man alle samples als Eintrag hat
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
    def read_from_file(cls, file, features, labels, **kwargs):
        df_samples = _utils.read_df_from_file(file)
        return cls(df_samples, features, labels, data_path=file,
                   **kwargs)

    @classmethod
    def read_from_yaml(cls, argsLoader, **kwargs) -> object:
        assert hasattr(argsLoader, 'source'), 'Define source of DataLoader! Can be either set to "load" or "create".'
        assert argsLoader.source in ['load', 'create'], 'Decide if source is "load" or "create"! "{}" not a valid ' \
                                                        'source'.format(argsLoader.source)

        if argsLoader.source == 'load':
            assert hasattr(argsLoader, 'load_DataLoader'), '"Load" source flag requires dict "load_DataLoader" with ' \
                                                           'following structure: \n{}'.format(
                TabularLoader.yaml_template(['load_DataLoader']))
            assert 'path' in argsLoader.load_DataLoader, 'Path in dict "load_DataLoader" is missing! DataLoader cannot be restored!'

            _, file_extention = os.path.splitext(argsLoader.load_DataLoader['path'])
            if file_extention == '.pkl':
                Loader = TabularLoader.load(argsLoader.load_DataLoader['path'])
                Loader.lparams.data_path = argsLoader.load_DataLoader['path']
            elif file_extention == '.ckpt':
                Loader = TabularLoader.read_from_checkpoint(argsLoader.load_DataLoader['path'])
            else:
                raise TypeError('Not supported file type to load DataLoader! Only supported are ".pkl" and ".ckpt"')

            if kwargs['batch'] is not None:
                Loader.lparams.batch = kwargs.pop('batch')
                logging.info('Batch size stored in file in overwritten by kwargs argument')
            if kwargs['num_workers'] is not None:
                Loader.lparams.num_workers = kwargs.pop('num_workers')
                logging.info('Num_workers stored in file in overwritten by kwargs argument')

        elif argsLoader.source == 'create':
            assert hasattr(argsLoader, 'create_DataLoader'), '"Create" source flag requires dict "create_DataLoader" ' \
                                                             'with following structure: \n{}'.format(
                TabularLoader.yaml_template(['create_DataLoader']))
            argsLoader = Namespace(**argsLoader.create_DataLoader)

            # create Loader
            assert hasattr(argsLoader, 'raw_data_path'), 'Argument "raw_data_path" missing! Define location of data to ' \
                                                         'construct DataLoader!'
            assert hasattr(argsLoader, 'features'), 'Argument "features" missing! Define features as list of str, e. g.' \
                                                    ' [PV, t, T_0]'
            assert hasattr(argsLoader, 'labels'), 'Argument "labels" missing! Define labels as list of str, e. g.' \
                                                  ' [yCO2, wCO]'
            Loader = TabularLoader.read_from_file(argsLoader.raw_data_path, features=argsLoader.features,
                                                  labels=argsLoader.labels, **kwargs)

            # validation data
            assert hasattr(argsLoader, 'validation_data'), 'Definition of validation data missing! Insert dict ' \
                                                           '"validation_data" with following structure \n{}'. \
                format(TabularLoader.yaml_template(['create_DataLoader', 'validation_data']))
            assert 'source' in argsLoader.validation_data, 'Define source of Validation data! Can be either set to ' \
                                                           '"load" or "split".'
            if argsLoader.validation_data['source'] == 'load':
                assert 'load_data' in argsLoader.validation_data, 'Parameters to perform data load are missing!'
                Loader.add_val_data(**argsLoader.validation_data['load_data'])
            elif argsLoader.validation_data['source'] == 'split':
                assert 'split_data' in argsLoader.validation_data, 'Parameters to perform data split are missing!'
                Loader.val_split(**argsLoader.validation_data['split_data'])
            else:
                raise TypeError('No validation data selected! Either set source flag to "load" or "split"')

            # test data
            assert hasattr(argsLoader, 'test_data'), 'Definition of test data missing! Insert dict "test_data" with ' \
                                                     'following structure \n{}'.\
                format(TabularLoader.yaml_template(['create_DataLoader', 'test_data']))
            assert 'source' in argsLoader.test_data, 'Define source of test data! Can be set to "load" or "split"'
            if argsLoader.test_data['source'] == 'load':
                assert 'load_data' in argsLoader.test_data, 'Parameters to perform data load are missing!'
                Loader.add_test_data(**argsLoader.test_data['load_data'])
            elif argsLoader.test_data['source'] == 'split':
                assert 'split_data' in argsLoader.test_data, 'Parameters to perform data split are missing!'
                Loader.test_split(**argsLoader.test_data['split_data'])
            else:
                raise TypeError('No test data selected! Either set source flag to "load" or "split"')

            # save loader
            if hasattr(argsLoader, 'save_Loader'):
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
        model = LightningFlexMLP.load_from_checkpoint(ckpt_file)  # TODO: implement for all model types

        assert hasattr(model.hparams, 'data_path'), 'Data cannot be reloaded because the pass is missing'
        _, file_extention = os.path.splitext(model.hparams.data_path)

        if file_extention == '.pkl':
            Loader = TabularLoader.load(model.hparams.data_path)
        else:
            assert all(hasattr(model.hparams, elem) for elem in ['features', 'labels', 'batch', 'num_workers',
                                                                 'x_scaler', 'y_scaler']), 'Parameters missing!'
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

    @staticmethod
    def yaml_template(key_list):
        template = {'type': 'TabularLoader',
                    'source': 'load/ create',
                    'load_DataLoader': {'path': 'name.pkl or modelXXX.ckpt'},
                    'create_DataLoader': {'raw_data_path': 'samples_name.csv, .txt, .h5, .flut',
                                          # TODO: change extension of flut datatype
                                          'features': ['feature_1', 'feature_2', '...'],
                                          'labels': ['label_1', 'label_2', '...'],
                                          'validation_data': {'source': 'load/ split',
                                                              'load_data': {
                                                                  'path': 'samples_name.csv, .txt, .h5, .flut'},
                                                              'split_data': {'method': 'random/ percentage/ explicit',
                                                                             'val_params': 'split_params'}},
                                          'test_data': {'source': 'load/ split',
                                                        'load_data': {'path': 'samples_name.csv, .txt, .h5, .flut'},
                                                        'split_data': {'method': 'random/ percentage/ explicit',
                                                                       'test_params': 'split_params'}},
                                          'save_Loader': {'execute': 'bool', 'path': 'name.pkl'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        print(yaml.dump(template, sort_keys=False))
