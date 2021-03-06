#######################################################################################################################
# TabularLoader
#######################################################################################################################

# import packages
import os
import torch
import yaml
import pandas as pd
import numpy as np
from sklearn import preprocessing
from argparse import Namespace
from typing import Union, Optional, List, Any, TypeVar, Type
from pathlib import Path, PosixPath

from thunder_torch import _logger
from thunder_torch.loader import _utils
from thunder_torch.loader.DataLoaderBase import DataLoaderBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch.utils.general import load_model_from_checkpoint
import thunder_torch as tt

TabularLoaderType = TypeVar('TabularLoaderType', bound='TabularLoader')


class TabularLoader(DataLoaderBase):
    """
    DataLoader for tabular data. All parameters used to construct the Loader are saved as Namespace object under the
    key lparams. At the point where the TabularLoader is used to generate PyTorch DataLoader as an input of the
    pl.Trainer class, the Loader is required to have training, validation and test data. If validation and test data
    have not been defined (either by loading seperate datafiles or by a split operation) the training data is randomly
    split in 73% training, 20% validation and 7% test data.

    Possibilities to create a TabularLoader object
    ----------------------------------------------
    - read from a file (possible dtypes: ".csv", ".txt", ".h5")
    - load a saved TabularLoader
    - restore from model checkpoint (NOTE: in order to restore a Loader using a ckpt, the lparams have to be saved
    in the hparams namespace of the model. Thereby, the hparams update fkt can be used. An example is found under
    thunder_torch.utils.utils_execute.get_dataLoader)
    """

    def __init__(self, df_samples: Optional[pd.DataFrame] = None, features: Optional[List[str]] = None,
                 labels: Optional[List[str]] = None,
                 train_samples: Optional[np.ndarray] = None, train_labels: Optional[np.ndarray] = None,
                 val_samples: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None,
                 test_samples: Optional[np.ndarray] = None, test_labels: Optional[np.ndarray] = None,
                 x_scaler: Optional[preprocessing.MinMaxScaler] = None,
                 y_scaler: Optional[preprocessing.MinMaxScaler] = None, batch: int = 64, num_workers: int = 10,
                 data_path: Optional[Union[str, Path, PosixPath]] = None, val_split: Optional[dict] = None,
                 val_path: Optional[dict] = None, test_split: Optional[dict] = None, test_path: Optional[dict] = None,
                 fastLoader: bool = False) -> None:
        """
        Create TabularLoader object

        Parameters
        ----------
        df_samples          - pd.DataFrame of samples
        features            - list of str: including features
        labels              - list of str: including labels
        train_samples       - np.ndarray containing the training samples
        val_samples         - np.ndarray containing the validation samples
        test_samples        - np.ndarray containing the testing samples
        x_scaler:           - sklearn.preprocessing MinMaxScaler for input samples
        y_scaler:           - sklearn.preprocessing MinMaxScaler for target samples
        batch:              - batch_size of the PyTorch DataLoader
        num_workers:        - num_workers of the PyTorch DataLoader
        data_path:          - Path to the training data used to create DataLoader
        val_split:          - dict including method and split_params to separate training samples in train and val set
        val_path:           - path to validation data
        test_split:         - dict including method and split_params to separate training samples in train and test set
        test_path:          - path to test data
        """
        super().__init__()

        self.lparams = Namespace()

        if all(elem is None for elem in [df_samples, features, labels]):
            self.x_train = train_samples
            self.y_train = train_labels
        else:
            self.lparams.features = features
            self.lparams.labels = labels
            self.x_train = df_samples[features]
            self.y_train = df_samples[labels]
            self.__check_lparams()

        self.lparams.x_scaler = x_scaler
        self.lparams.y_scaler = y_scaler
        self.lparams.batch = batch
        self.lparams.num_workers = num_workers
        self.lparams.data_path = data_path
        self.lparams.fast_loader = fastLoader
        self.get_scaler()

        if val_split:
            self.val_split(**val_split)
        elif val_path:
            self.add_val_data(**val_path)
        elif all(elem is not None for elem in [val_samples, val_labels]):
            self.x_val = val_samples
            self.y_val = val_labels

        if test_split:
            self.test_split(**test_split)
        elif test_path:
            self.add_test_data(**test_path)
        elif all(elem is not None for elem in [test_samples, test_labels]):
            self.x_test = test_samples
            self.y_test = test_labels

    def __check_lparams(self) -> None:
        assert all(isinstance(elem, str) for elem in self.lparams.features), "Given features is not a list of strings!"
        assert all(isinstance(elem, str) for elem in self.lparams.labels), "Given labels is not a list of strings!"
        assert all(elem not in self.lparams.labels for elem in self.lparams.features), "Feature is included in labels"

    # training_data ###################################################################################################
    def add_train_data(self, path: Union[Path, PosixPath, str], sep: str = ',') -> None:  # type: ignore[override]
        """
        Load training samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        sep              - separator
        """
        self.lparams.data_path = path
        self.lparams.sep = sep
        samples_train = _utils.read_df_from_file(path, sep)

        if self.x_train is not None:
            _logger.warning('Train data overwritten')

        self.x_train = samples_train[self.lparams.features]
        self.y_train = samples_train[self.lparams.labels]
        _logger.debug(f'Train samples added from file {path} with sep {sep}!')

    # validation_data #################################################################################################
    def add_val_data(self, path: Union[Path, PosixPath, str], sep: str = ',') -> None:  # type: ignore[override]
        """
        Load validation samples and separate them into input and target samples

        Parameters
        ----------
        val_samples     - file path
        sep             - separator
        """
        self.lparams.val = {'path': path, 'sep': sep}
        samples_val = _utils.read_df_from_file(path, sep)

        if self.x_val is not None:
            _logger.warning('Validation data overwritten')

        self.x_val = samples_val[self.lparams.features]
        self.y_val = samples_val[self.lparams.labels]
        _logger.debug(f'Validation samples added from file {path} with sep {sep}!')

    # test_data #######################################################################################################
    def add_test_data(self, path: Union[Path, PosixPath, str], sep: str = ',') -> None:  # type: ignore[override]
        """
        Load test samples and separate them into input and target samples

        Parameters
        ----------
        test_samples     - file path
        sep              - separator
        """
        self.lparams.test = {'path': path, 'sep': sep}
        samples_test = _utils.read_df_from_file(path, sep)

        if self.x_test is not None:
            _logger.warning('Test data overwritten')

        self.x_test = samples_test[self.lparams.features]
        self.y_test = samples_test[self.lparams.labels]
        _logger.debug(f'Test samples added from file {path} with sep {sep}!')

    # create pytorch dataloaders ######################################################################################
    def get_dataloader(self, x_samples: pd.DataFrame, y_samples: pd.DataFrame, **kwargs: Any) -> \
            Union[torch.utils.data.DataLoader, _utils.FastTensorDataLoader]:  # type: ignore[name-defined]
        if self.lparams.fast_loader:
            return _utils.FastTensorDataLoader(torch.tensor(x_samples), torch.tensor(y_samples),
                                               batch_size=self.lparams.batch, **kwargs)
        else:
            tensor = self.get_tensorDataset(x_samples, y_samples)
            return torch.utils.data.DataLoader(tensor, batch_size=self.lparams.batch,  # type: ignore[attr-defined]
                                               num_workers=self.lparams.num_workers, **kwargs)

    def train_dataloader(self, **kwargs: Any) -> \
            Union[torch.utils.data.DataLoader, _utils.FastTensorDataLoader]:  # type: ignore[name-defined]
        """
        Generate PyTorch DataLoader for the training data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_train, self.y_train)
        return self.get_dataloader(x_samples, y_samples, **kwargs)

    def val_dataloader(self, **kwargs: Any) -> \
            Union[torch.utils.data.DataLoader, _utils.FastTensorDataLoader]:  # type: ignore[name-defined]
        """
        Generate PyTorch DataLoader for the validation data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_val, self.y_val)
        return self.get_dataloader(x_samples, y_samples, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> \
            Union[torch.utils.data.DataLoader, _utils.FastTensorDataLoader]:  # type: ignore[name-defined]
        """
        Generate PyTorch DataLoader for the test data (all kwargs of the PyTorch DataLoader can be used)
        """
        x_samples, y_samples = self.data_normalization(self.x_test, self.y_test)
        return self.get_dataloader(x_samples, y_samples, **kwargs)

    # classmethods ####################################################################################################
    @classmethod
    def read_from_file(cls: Type[TabularLoaderType], file: Union[str, Path, PosixPath],   # type: ignore[override]
                       features: List[str], labels: List[str], **kwargs: Any) -> TabularLoaderType:
        """
        Create TabularLoader object from file

        Parameters
        ----------
        file            - file path
        features        - list of features
        labels          - list of labels
        kwargs          - see kwargs __init__ and additional:
            sep         - separator in the used datafile (for ulf always None, otherwise default ','

        Returns
        -------
        object          - TabularLoader object
        """
        sep = kwargs.pop('sep', ',')
        df_samples = _utils.read_df_from_file(file, sep)
        return cls(df_samples, features, labels, data_path=file, **kwargs)

    @classmethod
    def read_from_yaml(cls: Type[TabularLoaderType], argsLoader: dict, **kwargs: Any) -> TabularLoaderType:
        """
        Create TabularLoader object from a dict similar to the one given under yml_template

        Parameters
        ----------
        argsLoader      - arguments to create the Loader
        kwargs          - see kwargs __init__

        Returns
        -------
        object          - TabularLoader object
        """
        options = TabularLoader.__get_OptionClass()
        OptionClass.checker(input_dict=argsLoader, option_classes=options)

        if 'load_dataloader' in argsLoader:
            _, file_extention = os.path.splitext(argsLoader['load_dataloader']['path'])
            if file_extention == '.pkl':
                Loader: TabularLoader = TabularLoader.load(argsLoader['load_dataloader']['path'])
                Loader.lparams.data_path = argsLoader['load_dataloader']['path']
            elif file_extention == '.ckpt':
                Loader = TabularLoader.read_from_checkpoint(argsLoader['load_dataloader']['path'])
            else:
                raise TypeError('Not supported file type to load DataLoader! Only supported are ".pkl" and ".ckpt"')

            if kwargs.get('batch'):
                Loader.lparams.batch = kwargs.pop('batch')
                _logger.info('Batch size stored in file in overwritten by kwargs argument')
            if kwargs.get('num_workers'):
                Loader.lparams.num_workers = kwargs.pop('num_workers')
                _logger.info('Num_workers stored in file in overwritten by kwargs argument')

        elif 'create_dataloader' in argsLoader:
            argsCreate = argsLoader['create_dataloader']

            # create Loader
            Loader = TabularLoader.read_from_file(argsCreate.pop('raw_data_path'), features=argsCreate.pop('features'),
                                                  labels=argsCreate.pop('labels'), **kwargs)

            # validation data
            if 'validation_data' in argsCreate:
                if 'load_data' in argsCreate['validation_data']:
                    Loader.add_val_data(**argsCreate['validation_data']['load_data'])
                elif 'split_data' in argsCreate['validation_data']:
                    Loader.val_split(**argsCreate['validation_data']['split_data'])
                else:
                    raise KeyError('No validation data selected! Either include dict "load_data" or "split_data".')

            # test data
            if 'test_data' in argsCreate:
                if 'load_data' in argsCreate['test_data']:
                    Loader.add_test_data(**argsCreate['test_data']['load_data'])
                elif 'split_data' in argsCreate['test_data']:
                    Loader.test_split(**argsCreate['test_data']['split_data'])
                else:
                    raise KeyError('No test data selected! Either include dict "load_data" or "split_data".')

            # save loader
            if 'save_loader' in argsCreate:
                Loader.save(argsCreate['save_loader']['path'])

        else:
            raise KeyError('No DataLoader generated! Either include dict "load_DataLoader" or "create_DataLoader"!')

        return Loader  # type: ignore[return-value]

    @classmethod
    def read_from_checkpoint(cls: Type[TabularLoaderType],  # type: ignore[override]
                             ckpt_file: str) -> TabularLoaderType:
        """
        Create cls TabluarLoader from pytorch lightning checkpoint
        !! Hparams of the checkpoint had to be updated with lparams of the Loader in order to reconstruct the Loader!!

        Parameters
        ----------
        ckpt_file       - lightning checkpoint file

        Returns
        -------
        object          - TabularLoader object
        """
        pl_model = load_model_from_checkpoint(ckpt_file)
        lparams = pl_model.hparams.lparams

        assert hasattr(lparams, 'data_path'), 'Data cannot be reloaded because the pass is missing'
        _, file_extention = os.path.splitext(lparams.data_path)

        if file_extention == '.pkl':
            Loader = TabularLoader.load(lparams.data_path)
        else:
            assert all(hasattr(lparams, elem) for elem in ['features', 'labels', 'batch', 'num_workers',
                                                           'x_scaler', 'y_scaler']), 'Parameters missing!'
            Loader = cls.read_from_file(lparams.data_path, features=lparams.features, labels=lparams.labels,
                                        batch=lparams.batch, num_workers=lparams.num_workers, x_scaler=lparams.x_scaler,
                                        y_scaler=lparams.y_scaler)

        if 'path' in lparams.val:
            Loader.add_val_data(lparams.val.path, lparams.val.sep)
        elif all(elem in lparams.val for elem in ['method', 'params']):
            Loader.val_split(method=lparams.val['method'], params=lparams.val['params'])
        else:
            _logger.debug('NO validation data included!')

        if 'path' in lparams.test:
            Loader.add_test_data(lparams.test.path, lparams.test.sep)
        elif all(elem in lparams.test for elem in ['method', 'params']):
            Loader.test_split(method=lparams.test['method'], params=lparams.test['params'])
        else:
            _logger.debug('NO test data included!')

        return Loader  # type: ignore[return-value]

    @staticmethod
    def __get_OptionClass() -> dict:
        options = {'DataLoader': OptionClass(template=TabularLoader.yaml_template(['DataLoader']))}
        options['DataLoader'].add_key('type', dtype=str, required=True)
        options['DataLoader'].add_key('load_dataloader', dtype=dict, mutually_exclusive=['create_dataloader'])
        options['DataLoader'].add_key('create_dataloader', dtype=dict, mutually_exclusive=['load_dataloader'])

        options['load_dataloader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader',
                                                                                       'load_DataLoader']))
        options['load_dataloader'].add_key('path', dtype=[str, Path, PosixPath], required=True)

        options['create_dataloader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader',
                                                                                         'create_DataLoader']))
        options['create_dataloader'].add_key('raw_data_path', dtype=[str, Path, PosixPath], required=True)
        options['create_dataloader'].add_key('features', dtype=list, required=True)
        options['create_dataloader'].add_key('labels', dtype=list, required=True)
        options['create_dataloader'].add_key('validation_data', dtype=dict)
        options['create_dataloader'].add_key('test_data', dtype=dict)
        options['create_dataloader'].add_key('save_loader', dtype=dict)
        options['create_dataloader'].add_key('fast_loader', dtype=bool)

        options['validation_data'] = OptionClass(template=TabularLoader.yaml_template(
            ['DataLoader', 'create_DataLoader', 'validation_data']))
        options['validation_data'].add_key('load_data', dtype=dict, mutually_exclusive=['split_data'])
        options['validation_data'].add_key('split_data', dtype=dict, mutually_exclusive=['load_data'])
        options['test_data'] = options['validation_data']
        options['test_data'].template = TabularLoader.yaml_template(['DataLoader', 'create_DataLoader', 'test_data'])

        options['load_data'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                 'validation_data', 'load_data']))
        options['load_data'].add_key('path', dtype=[str, Path, PosixPath], required=True)
        options['load_data'].add_key('sep', dtype=str)

        options['split_data'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                  'validation_data', 'split_data']))
        options['split_data'].add_key('method', dtype=str, required=True)
        options['split_data'].add_key('params', dtype=[float, dict], required=True, param_dict=True)

        options['save_loader'] = OptionClass(template=TabularLoader.yaml_template(['DataLoader', 'create_DataLoader',
                                                                                   'save_Loader']))
        options['save_loader'].add_key('path', dtype=[str, Path, PosixPath], required=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template of a TabularLoader object
        """
        template = {'DataLoader': {'type': 'TabularLoader',
                                   '###INFO###': 'load_DataLoader and create_DataLoader mutually exclusive',
                                   'load_DataLoader': {'path': 'name.pkl or modelXXX.ckpt'},
                                   'create_DataLoader': {'raw_data_path': 'samples_name.csv, .txt, .h5, .ulf',
                                                         'features': ['feature_1', 'feature_2', '...'],
                                                         'labels': ['label_1', 'label_2', '...'],
                                                         'validation_data':
                                                             {'###INFO###': 'load_data & split_data mutually exclusive',
                                                              'load_data': {'path': 'samples_name.csv, .txt, .h5, .ulf',
                                                                            'sep': 'separator (default: ","'},
                                                              'split_data': {'method': 'random/ percentage/ explicit',
                                                                             'params': 'split_params'}},
                                                         'test_data':
                                                             {'###INFO###': 'load_data & split_data mutually exclusive',
                                                              'load_data': {'path': 'samples_name.csv, .txt, .h5, .ulf',
                                                                            'sep': 'separator (default: ","'},
                                                              'split_data': {'method': 'random/ percentage/ explicit',
                                                                             'params': 'split_params'}},
                                                         'save_Loader': {'path': 'name.pkl'},
                                                         'fast_loader': 'bool (default: False)'}}}

        template = tt.utils.get_by_path(template, key_list)

        return yaml.dump(template, sort_keys=False)
