# import packages
import argparse
import logging
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-yaml_path', type=str, default='input_MultiModelTraining.yaml',  # TODO: geaendert für debug
                        help='Name of yaml file to construct Neural Network')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help="Change logging level")
    return parser.parse_args()


def create_logger(argument):
    """
    Function to create the basic logger for the trainFlexNN scripts. Sets the logger level to the desired value (default
    is INFO, when run with -d it is DEBUG)

    Parameters
    ----------
    argument: is parsed

    Returns
    -------
    logger: logger handle pointing to sys.stdout on defined logger level

    """
    logger = logging.getLogger('trainFlexNN')
    if argument.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(module)s: %(message)s")
    else:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
