# import packages
import argparse
from git import HEAD
import torch
import importlib
from stfs_pytoolbox.ML_Utils import _logger
import logging
from stfs_pytoolbox.ML_Utils import _modules_models
from typing import Optional


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


def load_model_from_checkpoint(checkpoint_path: str):
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
    if not "model_type" in c["hparams"].keys():
        exit("ERROR in load_model_from_checkpoint: "
             "Cannot use this function since there is no 'model_type' key available in hparams.")

    model_type = c["hparams"]["model_type"]
    for m in _modules_models:
        try:
            model_class = getattr(importlib.import_module(m), model_type)
            _logger.debug(f'{model_type} fct found in {m}')
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug(f'{model_type} fct not found in {m}')
        assert False, f'{model_type} could not be found in {_modules_models}'

    return model_class.load_from_checkpoint(checkpoint_path)


def dynamic_imp(module_path: str, class_name: Optional[str] = None):
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
