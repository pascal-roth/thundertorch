# import packages
import argparse
from stfs_pytoolbox.ML_Utils import _logger
import logging


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-yaml_path', type=str, default='input_MultiModelTraining.yaml',
                        help='Name of yaml file to construct Neural Network')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help="Change logging level")
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
