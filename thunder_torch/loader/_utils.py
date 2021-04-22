#######################################################################################################################
# Utility function to generate DataLoaders
#######################################################################################################################

# import packages
import pandas as pd
import random
import numpy as np
import os
import torch
from typing import Union, Tuple
from pathlib import Path, PosixPath
from sklearn.model_selection import train_test_split

from thunder_torch import _logger


# data split ##########################################################################################################
def data_split_random(x_samples: pd.DataFrame, y_samples: pd.DataFrame, split_params: float) -> tuple:
    """
    Randomly split x_samples and y_samples DataFrame by a percentage

    Parameters
    ----------
    x_samples           - pd.DataFrame including the input samples
    y_samples           - pd.DataFrame including the target samples
    split_params        - split percentage

    Returns
    -------
    x_samples           - input samples reduced by split percentage
    x_split             - input samples taken from x_samples
    y_samples           - target samples reduced by split percentage
    y_split             - target samples taken from y_samples
    """
    assert isinstance(split_params, float), 'Val_size must be float in range 0 to 1!'
    assert split_params < 1, 'Percentage exceeds 100%!'
    x_samples, x_split, y_samples, y_split = train_test_split(x_samples, y_samples, test_size=split_params)
    _logger.info(f'Random split with percentage {split_params} has been performed!')

    return x_samples, x_split, y_samples, y_split


def data_split_percentage(x_samples: pd.DataFrame, y_samples: pd.DataFrame, split_params: dict) -> tuple:
    """
    Split the data by extracting the different values of a feature and randomly pick a certain percentage of it. All
    samples where the feature is equal to one of those values are extracted into x_split / y_split. However,
    if the feature has a different value for each sample, the method is equal to random. Furthermore, the size of
    x_split / y_split can differ from the percentage of values taken. In split_params the percentage can be defined
    for an arbitrary number of features.

    Parameters
    ----------
    x_samples           - pd.DataFrame including the input samples
    y_samples           - pd.DataFrame including the target samples
    split_params        - dict including the feature as key and the corresponding percentage as value
                          (e. g. {'feature_1': 0.2, 'feature_2': 0.05})

    Returns
    -------
    x_samples           - input samples reduced by split operation
    x_split             - input samples taken from x_samples
    y_samples           - target samples reduced by split operation
    y_split             - target samples taken from y_samples
    """
    assert isinstance(split_params, dict), 'split parameters have to be of type dict'
    x_split = pd.DataFrame([])

    for key, value in split_params.items():

        assert key in x_samples.columns, f'Feature "{key}" not included in the data features "{x_samples.columns}"'
        key_options = x_samples['{}'.format(key)].drop_duplicates()

        key_options = x_samples['{}'.format(key)].drop_duplicates()

        assert value < 1, 'Percentage exceeds 100%!'
        key_list = random.sample(list(key_options.values), k=int(np.round(value * len(key_options))))
        assert len(key_list) > 0, 'Percentage to low that one value of {} is selected'.format(key)

        x_samples = x_samples.rename(columns={'{}'.format(key): 'feature'})

        for i, key_value in enumerate(key_list):
            x_split = pd.concat((x_split, x_samples[x_samples.feature == key_value]), axis=0)
            x_samples = x_samples[x_samples.feature != key_value]

        x_samples = x_samples.rename(columns={'feature': '{}'.format(key)})
        x_split = x_split.rename(columns={'feature': '{}'.format(key)})

    y_split = y_samples[y_samples.index.isin(x_split.index)]
    y_samples = y_samples[y_samples.index.isin(x_samples.index)]

    _logger.info(f'Percentage split with params {split_params} has been performed! A percentage of '
                 f'{len(x_split)/(len(x_split) + len(x_samples))} samples has been separated.')

    return x_samples, x_split, y_samples, y_split


def data_split_explicit(x_samples: pd.DataFrame, y_samples: pd.DataFrame, split_params: dict) -> tuple:
    """
    Split data according to explicit values of the different features. It is possible to define an arbitrary number of
    values for the different features.

    Parameters
    ----------
    x_samples           - pd.DataFrame including the input samples
    y_samples           - pd.DataFrame including the target samples
    split_params        - dict including the feature as key and the corresponding explicit values
                          (e. g. {'feature_1': [value_1, value_2] , 'T': [740, 850, 1100]})

    Returns
    -------
    x_samples           - input samples reduced by split operation
    x_split             - input samples taken from x_samples
    y_samples           - target samples reduced by split operation
    y_split             - target samples taken from y_samples
    """
    assert isinstance(split_params, dict), 'Split parameters have to be of type dict'
    x_split = pd.DataFrame([])

    for key, value in split_params.items():

        assert key in x_samples.columns, f'Feature "{key}" not included in the data features "{x_samples.columns}"'
        key_options = x_samples['{}'.format(key)].drop_duplicates()

        if isinstance(value, (float, int)):
            key_list = [value]
        elif isinstance(value, list):
            key_list = value
        else:
            raise TypeError('type {} not valid for method "explicit", valid types are single or '
                            'list of float and int values!'.format(type(value)))

        x_samples = x_samples.rename(columns={'{}'.format(key): 'feature'})

        for i, key_value in enumerate(key_list):
            assert key_value in key_options.values, 'Value: {} is not included in feature {}'.format(value, key)
            x_split = pd.concat((x_split, x_samples[x_samples.feature == key_value]), axis=0)
            x_samples = x_samples[x_samples.feature != key_value]

        x_samples = x_samples.rename(columns={'feature': '{}'.format(key)})
        x_split = x_split.rename(columns={'feature': '{}'.format(key)})

    y_split = y_samples[y_samples.index.isin(x_split.index)]
    y_samples = y_samples[y_samples.index.isin(x_samples.index)]

    _logger.info(f'Explicit split with params {split_params} has been performed! A percentage of '
                 f'{len(x_split)/(len(x_split) + len(x_samples))} samples has been separated.')

    return x_samples, x_split, y_samples, y_split


# data loading ########################################################################################################
def read_df_from_file(file_path: Union[Path, PosixPath, str], sep: str = ',') -> pd.DataFrame:
    """
    Load samples of different data types

    Parameters
    ----------
    file_path           - sample path
    sep                 - seperator of the dataset

    Returns
    -------
    df_samples          - pd.DataFrame including samples
    """
    _, file_extention = os.path.splitext(file_path)

    if file_extention == '.h5':
        _logger.debug('"h5" datatype recognized')
        assert os.path.isfile(file_path), "Given h5-file '{}' doesn't exist.".format(file_path)
        with pd.HDFStore(file_path, 'r') as store:
            keys = store.keys()
            assert len(keys) == 1, "There must be only one key stored in pandas.HDFStore in '{}'!".format(file_path)
            df_samples = store.get(keys[0])

    elif file_extention == '.ulf':
        _logger.debug('"ulf" datatype recognized')
        sep_ulf = None
        with open(file_path, 'r') as f:
            output_dict = next(f).split(sep_ulf)
            data = np.vstack([np.array(line.split(sep_ulf), dtype=np.float64) for line in f])
        data = pd.DataFrame(data)
        data.columns = output_dict

    elif file_extention == '.csv' or file_extention == '.txt':
        _logger.debug('"csv"/ "txt" datatype recognized')
        df_samples = pd.read_csv(file_path, sep=sep)

    elif isinstance(file_path, pd.DataFrame):
        _logger.debug('"DataFrame" datatype recognized')
        df_samples = file_path

    else:
        raise TypeError('File of type: {} not supported!'.format(file_extention))

    assert df_samples.shape[1] != 1, 'Wrong separator chosen! All features loaded as one!'
    _logger.debug('Samples loaded successfully!')

    return df_samples


# Fast DataLoader used for Tabular Datasets ###########################################################################
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    Code-Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    """

    def __init__(self, *tensors: torch.Tensor, batch_size: int, shuffle: bool = False) -> None:
        """
        Initialize a FastTensorDataLoader
        Parameters
        ----------
        tensors         tensors to store. Must have the same length @ dim 0.
        batch_size      batch size to load.
        shuffle         if True, shuffle the data *in-place* whenever an iterator is created out of this object.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors: list = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self) -> None:
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        # return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        return self.n_batches
