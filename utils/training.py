#######################################################################################################################
# Util functions to execute a ML task (load/ create DataLoader, load/ create Model, start Training)
#######################################################################################################################

# import packages
import argparse
import importlib
import os
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils import _logger
from stfs_pytoolbox.ML_Utils import logger     # Logger that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import callbacks  # Callbacks that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import _modules_models, _modules_loader, _modules_callbacks


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

    for m in _modules_models:
        try:
            model_cls = getattr(importlib.import_module(m), argsModel.type)
            _logger.debug(f'Model Class of type {argsModel.type} has been loaded from {m}')
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug(f'Model Class of type {argsModel.type} has NOT been loaded from {m}')

    if hasattr(argsModel, 'load_model'):
        if os.path.isfile(argsModel.load_model['path']):
            model = model_cls.load_from_checkpoint(argsModel.load_model['path'])
            _logger.debug(f'Model successfully loaded from given file')

        elif os.path.isdir(argsModel.load_model['path']):
            checkpoints = []

            for file in os.listdir(argsModel.load_model['path']):
                if file.endswith(".ckpt"):
                    checkpoints.append(os.path.join(argsModel.load_model['path'], file))

            assert len(checkpoints) == 1, f'Either no or multiple checkpoint files are included in the given ' \
                                          f'directory: {argsModel.load_model["path"]}. Specify intended ckpt!'

            model = model_cls.load_from_checkpoint(checkpoints[0])
            _logger.debug(f'Model successfully loaded from given directory')
            
        else:
            raise AttributeError(f'Entered path {argsModel.load_model["path"]} does not exists!')

    elif hasattr(argsModel, 'create_model'):
        model = model_cls(argparse.Namespace(**argsModel.create_model))
        _logger.debug(f'Model successfully created')
    else:
        raise KeyError('Model not generated! Either include load_model or create_model dict!')

    if hasattr(argsModel, 'params'):
        model.hparams_update(update_dict=argsModel.params)
        _logger.debug('model default hparams updated by argsModel.params')
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
    for m in _modules_loader:
        try:
            loader_cls = getattr(importlib.import_module(m), argsLoader['type'])
            _logger.debug('Model Class of type {} has been loaded from {}'.format(argsLoader['type'], m))
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug('Model Class of type {} has NOT been loaded from {}'.format(argsLoader['type'], m))

    if model:
        dataLoader = loader_cls.read_from_yaml(argsLoader, batch=model.hparams.batch,
                                               num_workers=model.hparams.num_workers)
        model.hparams_update(update_dict={'lparams': dataLoader.lparams})
        _logger.info('DataLoader generated using batch_size and num_workers from model. Loader params are included '
                     'in model.hparams')
    else:
        dataLoader = loader_cls.read_from_yaml(argsLoader)
        _logger.info('DataLoader generated without model information and Loader params not included in model')

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
                # Check from which destination the callback class is loaded
                for m in _modules_callbacks:
                    try:
                        callback_cls = getattr(importlib.import_module(m), argsTrainer.callbacks[i]['type'])
                        break
                    except AttributeError:
                        _logger.debug('Callback of type {} NOT found in {}'.format(argsTrainer.callbacks[i]['type'], m))

                if 'params' in argsTrainer.callbacks[i]:
                    callback = callback_cls(**argsTrainer.callbacks[i]['params'])
                else:
                    callback = callback_cls()
                callback_list.append(callback)

        if callback_list != []:
            argsTrainer.params['callbacks'] = callback_list

    else:
        _logger.info('No callbacks implemented')

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
        _logger.info('No logger selected')

    # define trainer and start training
    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**argsTrainer.params))

    if all(getattr(dataLoader, item) is not None for item in ['x_val', 'y_val']):
        _logger.debug('Training and validation data included in DataLoader -> Model validation is performed!')
        trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    else:
        _logger.debug('NO validation data included in DataLoader -> Model validation is NOT performed!')
        trainer.fit(model, train_dataloader=dataLoader.train_dataloader())

    if all(getattr(dataLoader, item) is not None for item in ['x_test', 'y_test']):
        _logger.debug('Test data included in DataLoader -> Model testing performed!')
        trainer.test(model, test_dataloaders=dataLoader.test_dataloader())
    else:
        _logger.debug('NO test data included in DataLoader -> Model testing is NOT performed!')

    _logger.debug('MODEL TRAINING DONE')