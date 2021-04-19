# import packages
import argparse
import torch
import importlib
from thunder_torch import _logger
import logging
from thunder_torch import _modules_models
from typing import Optional, Union
from more_itertools import chunked
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning import LightningModule


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, default='template.yaml',
                        help='Name of yaml file to construct Neural Network')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help="Change logging level to debug")
    return parser.parse_args()


def logger_level(argument: argparse.Namespace) -> None:
    """
    Function to change "ml_utils" logger level to debug

    Parameters
    ----------
    argument: is parsed

    Returns
    -------
    logger: logger handle pointing to sys.stdout on defined logger level

    """
    if argument.debug:
        _logger.setLevel(logging.DEBUG)


def load_model_from_checkpoint(checkpoint_path: str) -> LightningModule:
    """
    Loads a model from a given checkpoint path even if the model class is not known prior by the code
    Unfortunately, this cannot be implemented in the ModelBase class without a lot of effort and therefore
    an external function is a much simpler solution.

    Parameters
    ----------
    checkpoint_path: str
        path to the checkpoint file

    Returns
    -------
    model_class:
        loaded model based off 'model_type' hyperparameter of checkpoint
    """

    c = torch.load(checkpoint_path, torch.device("cpu"))
    if "model_type" not in c["hparams"].keys():
        exit("ERROR in load_model_from_checkpoint: "
             "Cannot use this function since there is no 'model_type' key available in hparams.")

    model_type = c["hparams"]["model_type"]
    model_class: object

    for m in _modules_models:
        try:
            _, model_class = dynamic_imp(m, model_type)
            # model_class = getattr(importlib.import_module(m), model_type)
            _logger.debug(f'{model_type} fct found in {m}')
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug(f'{model_type} fct not found in {m}')
        # assert False, f'{model_type} could not be found in {_modules_models}'

    return model_class.load_from_checkpoint(checkpoint_path)


def dynamic_imp(module_path: str, class_name: Optional[str] = None) -> tuple:
    """

    Helper function to dynamically import class from custom modules.
    Adapted from: https://www.geeksforgeeks.org/how-to-dynamically-load-modules-or-classes-in-python/

    Parameters
    ----------
    module_path: str
        path to the file of the module
    class_name: Optional(str)
        name of the class that is to be imported

    Returns
    -------
    mypackage
        imported package
    myclass: class
        import class

    """
    import imp
    myclass = None
    mypackage = None
    # find_module() method is used
    # to find the module and return
    # its description and path
    try:
        mypackage = importlib.import_module(module_path)
        if class_name:
            myclass = getattr(mypackage, class_name)

    except ModuleNotFoundError:
        _logger.debug(f"Module '{module_path}' could not be imported, trying imp import")
        try:
            fp, path, desc = imp.find_module(module_path)
            # load_modules loads the module
            # dynamically ans takes the filepath
            # module and description as parameter
            mypackage = imp.load_module(module_path, fp, path, desc)
            if myclass:
                myclass = imp.load_module(f"{module_path}.{class_name}", fp, path, desc)

        except ImportError:
            raise ImportError(f"Neither importlib nor imp could not load '{class_name}' from '{module_path}'")

    return mypackage, myclass


def run_model(data: pd.DataFrame, checkpoint: Union[str, LightningModule], batch: int = 1000, noise_index: int = 0,
              noise: Optional[float] = None) -> pd.DataFrame:
    """
    Function to run a model created by this toolbox

    Parameters
    ----------
    data: DataFrame from which the input data for the model is taken
    checkpoint: checkpoint path of model or with the toolbox created LightningModel
    batch: Batch size to handle large inputs
    noise_index: index of input on which noise will be applied
    noise: amount of noise to apply

    Returns
    -------
    prediction: DataFrame with inference result

    """
    def to_tensor(data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data.astype(np.float64))

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def scale_input(X: np.ndarray, featureScaler: MinMaxScaler) -> np.ndarray:
        return featureScaler.transform(X)

    def scale_output(Y: np.ndarray, labelScaler: MinMaxScaler) -> np.ndarray:
        return labelScaler.inverse_transform(Y)

    model: LightningModule
    if isinstance(checkpoint, str):
        model = load_model_from_checkpoint(checkpoint)
    else:
        model = checkpoint

    features = model.hparams.lparams.features
    labels = model.hparams.lparams.labels
    featureScaler = model.hparams.lparams.x_scaler
    labelScaler = model.hparams.lparams.y_scaler

    df = data[features+labels].copy()
    index_chunks = chunked(df.index, batch)

    for ii in tqdm(index_chunks):
        x = scale_input(df.loc[ii, features].values, featureScaler)
        if noise is not None:
            assert noise_index <= len(features), "Noise index must be lower than number of features"
            gaussian_noise = np.random.normal(1, noise, len(x))
            x[:, noise_index] = np.multiply(x[:, noise_index], gaussian_noise)
            df.loc[ii, features[noise_index]+"_noise"] = x[:, noise_index]
            df.loc[ii, "noise"] = noise

        x = to_tensor(x)
        y = model(x)
        df.loc[ii, labels] = scale_output(to_numpy(y), labelScaler)
    return df
