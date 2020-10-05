#######################################################################################################################
# Load arguments of flexMLP_single.yaml and execute flexMLP_pl.py
#######################################################################################################################

# import packages
import yaml
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger

from argparse import Namespace

from flexMLP_pl import flexMLP_pl
from flexCNN_pl import flexCNN_pl
from TabularLoader import TabularLoader


def parse_yaml():
    flexMLP_yaml = open("flexMLP_single.yaml")
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def LoaderGenerator(argsLoader):
    """
    Generate TabularLoader object

    Parameters
    ----------
    argsLoader      - Namespace object with arguments related to create TabluarLoader object

    Returns
    -------
    Loader          - TabularLoader object including training, validation and test samples
    """

    _, file_exention = os.path.splitext(argsLoader.data)

    if file_exention == '.h5':
        Loader = TabularLoader.read_from_h5(argsLoader.data)
    elif file_exention == '.ulf':
        Loader = TabularLoader.read_from_flut(argsLoader.data)
    elif file_exention == '.csv':
        Loader = TabularLoader.read_from_csv(argsLoader.data)
    else:
        raise SyntaxError('Data type not supported! Supported are .csv, .txt, .ulf, .h5 with space discriminator')

    if argsLoader.validation_data['load_data']['perform']:
        Loader.add_val_data(argsLoader.validation_data['load_data']['location'])
    elif argsLoader.validation_data['split_data']['perform']:
        argsVal = Namespace(**argsLoader.validation_data['split_data'])
        Loader.val_split(method=argsVal.method, val_size=argsVal.val_size, split_method=argsVal.split_method,
                         val_params=argsVal.val_params)
    else:
        raise SyntaxError('No validation data selected! Either set perform flag in load or split data to "True".')

    if argsLoader.test_data['load_data']['perform']:
        Loader.add_test_data(argsLoader.test_data['load_data']['location'])
    elif argsLoader.test_data['split_data']['perform']:
        argsTest = Namespace(**argsLoader.test_data['split_data'])
        Loader.test_split(method=argsTest.method, test_size=argsTest.test_size, split_method=argsTest.split_method,
                          test_params=argsTest.test_params)
    else:
        raise SyntaxError('No test data selected! Either set perform flag in load or split data to "True".')

    return Loader


def main(argsLoader, argsModel, argsTrainer):
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    argsLoader      - Namespace object with arguments related to create TabluarLoader object
    argsModel       - Namespace object with arguments related to load/ create LightningModule
    argsTrainer     - Namespace object with arguments realted to training of LightningModule
    """
    Loader = LoaderGenerator(argsLoader)

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=12, verbose=False, mode='min')  # TODO: Discuss with Julian if necessary
    lr_logger = LearningRateLogger()

    # load/ create model
    if argsModel.load_model['perform']:
        model = flexMLP_pl.load_from_checkpoint(argsModel.load_model['location'], TabularLoader=Loader)
        trainer = pl.Trainer.from_argparse_args(argsTrainer, resume_from_checkpoint=argsModel.load_model['location'],
                                                early_stop_callback=early_stop_callback, callbacks=[lr_logger])
        trainer.fit(model)

    elif argsModel.create_model['perform']:
        hparams = Namespace(**argsModel.create_model)
        hparams.x_scaler = None
        hparams.y_scaler = None

        # check hyperparameters
        # TODO: was macht lightning, falls ich einen ungültigen Wert eingebe --> vllt diese Errors nicht nötig
        assert all(isinstance(elem, str) for elem in hparams.features), "Given features is not a list of strings!"
        assert all(isinstance(elem, str) for elem in hparams.labels), "Given labels is not a list of strings!"

        model = flexMLP_pl(hparams=hparams, TabularLoader=Loader)
        if hparams.scheduler:
            trainer = pl.Trainer.from_argparse_args(argsTrainer, early_stop_callback=early_stop_callback,
                                                    callbacks=[lr_logger])  # TODO: are callbacks saved by the checkpoint?
        else:
            trainer = pl.Trainer.from_argparse_args(argsTrainer, early_stop_callback=early_stop_callback)
        trainer.fit(model)

    else:
        raise SyntaxError('Model neither loaded nor created! Change perform flag in load/ create model to True.')

if __name__ == '__main__':
    args_yaml = parse_yaml()

    argsLoader = Namespace(**args_yaml['TabularLoader'])
    argsModel = Namespace(**args_yaml['flexMLP_pl'])
    argsTrainer = Namespace(**args_yaml['pl.Trainer'])

    main(argsLoader, argsModel, argsTrainer)