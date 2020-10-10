#######################################################################################################################
# Utility function to generate DataLoaders
#######################################################################################################################

# import packages
import pandas as pd
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split


# data split ######################################################################################################
def data_split_random(x_samples, y_samples, split_params):  #  -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    assert isinstance(split_params, float), 'Val_size must be float in range 0 to 1!'
    assert split_params < 1, 'Percentage exceeds 100%!'
    x_samples, x_split, y_samples, y_split = train_test_split(x_samples, y_samples, test_size=split_params)
    return x_samples, x_split, y_samples, y_split


def data_split_percentage(x_samples, y_samples, split_params):  #  -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    assert isinstance(split_params, dict), 'split parameters have to be of type dict'
    x_split = pd.DataFrame([])

    for key, value in split_params.items():

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

    return x_samples, x_split, y_samples, y_split


def data_split_explicit(x_samples, y_samples, split_params):  #  -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    assert isinstance(split_params, dict), 'Split parameters have to be of type dict'
    x_split = pd.DataFrame([])

    for key, value in split_params.items():

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

    return x_samples, x_split, y_samples, y_split


# data loading ########################################################################################################
def read_df_from_file(file_path) -> pd.DataFrame:
    """
    Load samples of different sample tpyes
    Parameters
    ----------
    file_path            - sample path

    Returns
    -------
    df_samples      - pd.DataFrame including samples
    """
    _, file_extention = os.path.splitext(file_path)
    # TODO: check delimiter (online function finden)
    if file_extention == '.h5':
        assert os.path.isfile(file_path), "Given h5-file '{}' doesn't exist.".format(file_path)
        store = pd.HDFStore(file_path)
        keys = store.keys()
        assert len(keys) == 1, "There must be only one key stored in pandas.HDFStore in '{}'!".format(file_path)
        df_samples = store.get(keys[0])
        store.close()
    elif file_extention == '.flut':
        # import pyflut  # TODO: nur laden when package available --> import ... wenn nicht geladen, fehler!
        raise NotImplementedError('not implemented yet -> flut datatype unknown')  # TODO: implement pyflut datatype
    elif file_extention == '.csv' or file_extention == '.txt':
        df_samples = pd.read_csv(file_path)
    elif isinstance(file_path, pd.DataFrame):
        df_samples = file_path
    else:
        raise TypeError('File of type: {} not supported!'.format(file_extention))

    return df_samples
