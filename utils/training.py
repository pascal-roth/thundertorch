#######################################################################################################################
# Util functions to execute a ML task (load/ create DataLoader, load/ create Model, start Training)
#######################################################################################################################

# import packages
import argparse
import logging
import importlib
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils import models     # Models that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import loader     # Loader that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import logger     # Logger that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import callbacks  # Callbacks that are defined in __all__ in the __init__ file


def get_model(argsModel) -> pl.LightningModule:
    """
    Load/ create the model given the model arguments

    Parameters
    ----------
    argsModel       - model arguments either as dict or Namespace object

    Returns
    -------
    model           - LightningModule
    """
    if isinstance(argsModel, dict):
        argsModel = argparse.Namespace(**argsModel)

    if hasattr(argsModel, 'load_model'):
        model = getattr(models, argsModel.type).load_from_checkpoint(argsModel.load_model['path'])
        logging.debug('Model has been loaded')
    elif hasattr(argsModel, 'create_model'):
        model = getattr(models, argsModel.type)(argparse.Namespace(**argsModel.create_model))
        logging.debug('Model has been created')
    elif hasattr(argsModel, 'import_model'):
        model = importlib.import_module(argsModel.type)
    else:
        raise KeyError('Model not generated! Either include load_model or create_model dict!')

    if hasattr(argsModel, 'params'):
        model.hparams_update(update_dict=argsModel.params)
        logging.debug('model default hparams updated by argsModel.params')
    return model.double()


def get_dataLoader(argsLoader: dict, model: pl.LightningModule = None):
    """
    Load/ create DataLoader object

    Parameters
    ----------
    argsLoader      - loader arguments
    model           - LightningModule that includes batch_size and num_workers parameters that are necessary for the
                      PyTorch DataLoaders as pl.Trainer input

    Returns
    -------
    dataLoader
    """
    if model:
        dataLoader = getattr(loader, argsLoader['type']).read_from_yaml(argsLoader, batch=model.hparams.batch,
                                                                        num_workers=model.hparams.num_workers)
        model.hparams_update(update_dict={'lparams': dataLoader.lparams})
        logging.info('DataLoader generated using batch_size and num_workers from model. Loader params are included '
                     'in model.hparams')
    else:
        dataLoader = getattr(loader, argsLoader['type']).read_from_yaml(argsLoader)
        logging.info('DataLoader generated without model information and Loader params not included in model')

    return dataLoader


def train_model(model: pl.LightningModule, dataLoader, argsTrainer) -> None:
    """
    Train a given model with the data included in the DataLoader object

    Parameters
    ----------
    model           - LightningModule
    dataLoader      - DataLoader object including training, validation and test dataset
    argsTrainer     - Trainer arguments
    """
    if isinstance(argsTrainer, dict):
        argsTrainer = argparse.Namespace(**argsTrainer)

    # create callback objects
    if hasattr(argsTrainer, 'callbacks'):
        callback_list = []
        if not isinstance(argsTrainer.callbacks, list): argsTrainer.callbacks = list(argsTrainer.callbacks)

        for i in range(len(argsTrainer.callbacks)):

            if argsTrainer.callbacks[i]['type'] == 'EarlyStopping':
                earlyStopping = pl.callbacks.EarlyStopping(**argsTrainer.callbacks[i]['params'])
                argsTrainer.params['early_stop_callback'] = earlyStopping
            elif argsTrainer.callbacks[i]['type'] == 'Checkpointing':
                checkpoint = callbacks.Checkpointing(**argsTrainer.callbacks[i]['params'])
                argsTrainer.params['checkpoint_callback'] = checkpoint
            else:
                if 'params' in argsTrainer.callbacks[i]:
                    callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])(
                        **argsTrainer.callbacks[i]['params'])
                else:
                    callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])()
                callback_list.append(callback)

        if callback_list != []:
            argsTrainer.params['callbacks'] = callback_list

    else:
        logging.info('No callbacks implemented')

    # create logger_fn objects
    if hasattr(argsTrainer, 'logger'):
        loggers = []
        if not isinstance(argsTrainer.logger, list): argsTrainer.logger = list(argsTrainer.logger)

        for i in range(len(argsTrainer.logger)):

            if argsTrainer.logger[i]['type'] == 'comet-ml':
                from pytorch_lightning.loggers.comet import CometLogger
                logger_fn = CometLogger(**argsTrainer.logger[i]['params'])
            elif argsTrainer.logger[i]['type'] == 'tensorboard':
                logger_fn = logger.TensorBoardLoggerAdjusted
            else:
                raise ValueError('Selected logger not implemented!')

            loggers.append(logger_fn)

        argsTrainer.params['logger'] = loggers

    else:
        argsTrainer.params['logger'] = False
        logging.info('No logger selected')

    # define trainer and start training
    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**argsTrainer.params))
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())