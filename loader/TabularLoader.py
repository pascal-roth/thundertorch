#######################################################################################################################
# TabularLoader
#######################################################################################################################

# import packages
import os
import torch
import pickle
import logging
import yaml
from sklearn import preprocessing
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.loader import _utils
from stfs_pytoolbox.ML_Utils.utils.utils_option_class import OptionClass


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
        self.lparams.val = {'path': path}
        samples_val = _utils.read_df_from_file(path)
        if self.x_val is not None: logging.warning('Validation data overwritten')
        self.x_val = samples_val[self.lparams.features]
        self.y_val = samples_val[self.lparams.labels]

    def val_split(self, **kwargs) -> None:
        """
        Split available samples into training and validation set
        """
        self.lparams.val = {'method': kwargs.pop('method', 'random'), 'params': kwargs.pop('params', 0.2)}

        self.x_train, self.x_val, self.y_train, self.y_val = getattr(_utils, 'data_split_' + self.lparams.val['method']) \
            (self.x_train, self.y_train, self.lparams.val['params'])

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "params"')

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
        if self.x_test is not None: logging.warning('Test data overwritten')
        self.x_test = samples_test[self.lparams.features]
        self.y_test = samples_test[self.lparams.labels]

    def test_split(self, **kwargs) -> None:
        """
        Split available samples into training and test set
        """
        self.lparams.test = {'method': kwargs.pop('method', 'random'), 'params': kwargs.pop('params', 0.2)}

        self.x_train, self.x_test, self.y_train, self.y_test = getattr(_utils, 'data_split_' + self.lparams.test['method']) \
            (self.x_train, self.y_train, self.lparams.test['params'])

        if len(kwargs) != 0:
            logging.warning('Additional, unexpected kwargs are given! Only expected args are: "method", "params"')

    # create pytorch dataloaders ######################################################################################
    def train_dataloader(self, **kwargs):
        if self.x_val is None: self.val_split()  # TODO: maybe find a better solution to add an default
        if self.x_test is None: self.test_split()

        self.x_train = self.lparams.x_scaler.transform(self.x_train)
        self.y_train = self.lparams.y_scaler.transform(self.y_train)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_train), torch.tensor(self.y_train))
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def val_dataloader(self, **kwargs):
        self.x_val = self.lparams.x_scaler.transform(self.x_val)
        self.y_val = self.lparams.y_scaler.transform(self.y_val)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_val), torch.tensor(self.y_val))
        return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch, num_workers=self.lparams.num_workers,
                                           **kwargs)

    def test_dataloader(self, **kwargs):
        assert self.x_test is not None, 'Test data has to be assigned before test_dataloader is created'  # TODO: schauen ob dann default genommen werden kann, wenn man alle samples als Eintrag hat
        self.x_test = self.lparams.x_scaler.transform(self.x_test)
        self.y_test = self.lparams.y_scaler.transform(self.y_test)
        tensor = torch.utils.data.TensorDataset(torch.tensor(self.x_test), torch.tensor(self.y_test))
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
        options = TabularLoader.get_OptionClass()
        OptionClass.checker(input_dict=argsLoader, option_classes=options)

        if 'load_DataLoader' in argsLoader:
            _, file_extention = os.path.splitext(argsLoader['load_DataLoader']['path'])
            if file_extention == '.pkl':
                Loader = TabularLoader.load(argsLoader['load_DataLoader']['path'])
                Loader.lparams.data_path = argsLoader['load_DataLoader']['path']
            elif file_extention == '.ckpt':
                Loader = TabularLoader.read_from_checkpoint(argsLoader['load_DataLoader']['path'])
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
            Loader = TabularLoader.read_from_file(argsCreate.pop('raw_data_path'), features=argsCreate.pop('features'),
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
        Create cls TabluarLoader from pytorch lightning checkpoint
        !! Hparams of the checkpoint had to be updated with lparams of the Loader in order to reconstruct the Loader!!

        Parameters
        ----------
        ckpt_file       - lightning checkpoint file

        Returns
        -------
        Loader          - TabularLoader class object
        """
        from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP

        model = LightningFlexMLP.load_from_checkpoint(ckpt_file)  # TODO: implement for all model types
        lparams = model.hparams.lparams

        assert hasattr(lparams, 'data_path'), 'Data cannot be reloaded because the pass is missing'
        _, file_extention = os.path.splitext(lparams.data_path)

        if file_extention == '.pkl':
            Loader = TabularLoader.load(lparams.data_path)
        else:
            assert all(hasattr(lparams, elem) for elem in ['features', 'labels', 'batch', 'num_workers',
                                                           'x_scaler', 'y_scaler']), 'Parameters missing!'
            Loader = TabularLoader.read_from_file(lparams.data_path, features=lparams.features,
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
    def get_OptionClass():
        options = {'DataLoader': OptionClass(template=TabularLoader.yaml_template(['DataLoader']))}
        options['DataLoader'].add_key('type', dtype=str, required=True)
        options['DataLoader'].add_key('load_DataLoader', dtype=dict, mutually_exclusive=['create_DataLoader'])
        options['DataLoader'].add_key('create_DataLoader', dtype=dict, mutually_exclusive=['load_DataLoader'])

        options['load_DataLoader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'load_DataLoader']))
        options['load_DataLoader'].add_key('path', dtype=str, required=True)

        options['create_DataLoader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader']))
        options['create_DataLoader'].add_key('raw_data_path', dtype=str, required=True)
        options['create_DataLoader'].add_key('features', dtype=list, required=True)
        options['create_DataLoader'].add_key('labels', dtype=list, required=True)
        options['create_DataLoader'].add_key('validation_data', dtype=dict, required=True)
        options['create_DataLoader'].add_key('test_data', dtype=dict, required=True)
        options['create_DataLoader'].add_key('save_Loader', dtype=dict)

        options['validation_data'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                       'validation_data']))
        options['validation_data'].add_key('load_data', dtype=dict, mutually_exclusive=['split_data'])
        options['validation_data'].add_key('split_data', dtype=dict, mutually_exclusive=['load_data'])
        options['test_data'] = options['validation_data']
        options['test_data'].template = TabularLoader.yaml_template(['DataLoader', 'create_DataLoader', 'test_data'])

        options['load_data'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                 'validation_data', 'load_data']))
        options['load_data'].add_key('path', dtype=str, required=True)

        options['split_data'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                  'validation_data', 'split_data']))
        options['split_data'].add_key('method', dtype=str, required=True)
        options['split_data'].add_key('params', dtype=[float, dict], required=True, param_dict=True)

        options['save_Loader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                   'save_Loader']))
        options['save_Loader'].add_key('path', dtype=str, required=True)

        return options

    @staticmethod
    def yaml_template(key_list):
        template = {'DataLoader': {'type': 'TabularLoader',
                                   'load_DataLoader': {'path': 'name.pkl or modelXXX.ckpt'},
                                   'create_DataLoader': {'raw_data_path': 'samples_name.csv, .txt, .h5, .flut',
                                                         # TODO: change extension of flut datatype
                                                         'features': ['feature_1', 'feature_2', '...'],
                                                         'labels': ['label_1', 'label_2', '...'],
                                                         'validation_data': {'load_data': {
                                                             'path': 'samples_name.csv, .txt, .h5, .flut'},
                                                                             'split_data': {
                                                                                 'method': 'random/ percentage/ explicit',
                                                                                 'params': 'split_params'}},
                                                         'test_data': {'load_data': {
                                                             'path': 'samples_name.csv, .txt, .h5, .flut'},
                                                                       'split_data': {
                                                                           'method': 'random/ percentage/ explicit',
                                                                           'params': 'split_params'}},
                                                         'save_Loader': {'execute': 'bool', 'path': 'name.pkl'}}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
