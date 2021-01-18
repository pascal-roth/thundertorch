#######################################################################################################################
# Util functions to execute a ML task (load/ create DataLoader, load/ create Model, start Training)
#######################################################################################################################

# import packages
import argparse
import importlib
import os
import pytorch_lightning as pl
import torch

from stfs_pytoolbox.ML_Utils import _logger
from stfs_pytoolbox.ML_Utils import logger  # Logger that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import callbacks  # Callbacks that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import _modules_models, _modules_loader, _modules_callbacks, _modules_loss, \
    _modules_optim, _modules_activation, _modules_lr_scheduler


def train_config(argsConfig: dict, argsTrainer: dict) -> dict:
    # add source path for modules defined in __init__
    if 'source_files' in argsConfig:
        # source_path = os.path.join(os.getcwd(), argsConfig['source_files'] + '.py')
        source_path = argsConfig['source_files']

        try:
            importlib.import_module(source_path)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Defined path is not a module!')

        if source_path in _modules_models:
            _logger.debug(f'Individual Module {source_path} already included')
        else:
            _modules_models.append(source_path)
            _modules_callbacks.append(source_path)
            _modules_optim.append(source_path)
            _modules_loss.append(source_path)
            _modules_activation.append(source_path)
            _modules_lr_scheduler.append(source_path)
            _modules_loader.append(source_path)
            _logger.debug(f'Individual Module {source_path} added')

    # check for reproducibility https://pytorch.org/docs/stable/notes/randomness.html
    if 'reproducibility' in argsConfig and argsConfig['reproducibility'] is True:
        pl.seed_everything(42)
        argsTrainer['params']['deterministic'] = True

    return argsTrainer


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


def train_model(model: pl.LightningModule, dataLoader, argsTrainer: dict) -> None:
    """
    Train a given model with the data included in the DataLoader object

    Parameters
    ----------
    model           - LightningModule
    dataLoader      - DataLoader object including training, validation and test dataset
    argsTrainer     - Trainer arguments
    """
    # create callback objects
    if 'callbacks' in argsTrainer:
        argsTrainer = train_callbacks(argsTrainer)
        _logger.debug(f'Callbacks added: {argsTrainer["callbacks"]}')
    else:
        _logger.debug('No callbacks implemented')

    # create logger_fn objects
    if 'logger' in argsTrainer:
        argsTrainer['params']['logger'] = train_logger(argsTrainer)
        _logger.debug(f'Training logger added: {argsTrainer["params"]["logger"]}')
    else:
        argsTrainer['params']['logger'] = False
        _logger.debug('No logger selected')

    # define trainer and start training, testing
    trainer = pl.Trainer(**argsTrainer['params'])
    execute_training(model, dataLoader, trainer)
    execute_testing(model, dataLoader, trainer)
    _logger.debug('MODEL TRAINING DONE')


def train_callbacks(argsTrainer: dict) -> dict:
    callback_list = []

    if not isinstance(argsTrainer['callbacks'], list):
        argsTrainer['callbacks'] = list(argsTrainer['callbacks'])

    for i in range(len(argsTrainer['callbacks'])):

        # Extra handling of EarlyStopping and Checkpointing callbacks because they have extra flags in the Trainer
        if argsTrainer['callbacks'][i]['type'] == 'EarlyStopping':
            earlyStopping = pl.callbacks.EarlyStopping(**argsTrainer['callbacks'][i]['params'])
            argsTrainer['params']['early_stop_callback'] = earlyStopping
        elif argsTrainer['callbacks'][i]['type'] == 'Checkpointing':
            checkpoint = callbacks.Checkpointing(**argsTrainer['callbacks'][i]['params'])
            argsTrainer['params']['checkpoint_callback'] = checkpoint
        else:
            # Check from which destination the callback class is loaded
            for m in _modules_callbacks:
                try:
                    callback_cls = getattr(importlib.import_module(m), argsTrainer['callbacks'][i]['type'])
                    break
                except AttributeError:
                    _logger.debug('Callback of type {} NOT found in {}'.format(argsTrainer['callbacks'][i]['type'], m))

            if 'params' in argsTrainer['callbacks'][i]:
                callback = callback_cls(**argsTrainer['callbacks'][i]['params'])
            else:
                callback = callback_cls()
            callback_list.append(callback)

    if callback_list:
        argsTrainer['params']['callbacks'] = callback_list
    else:
        argsTrainer['params']['callbacks'] = []

    return argsTrainer


def train_logger(argsTrainer: dict) -> list:
    loggers = []
    if not isinstance(argsTrainer['logger'], list):
        argsTrainer['logger'] = list(argsTrainer['logger'])

    for i in range(len(argsTrainer['logger'])):

        assert 'params' in argsTrainer['logger'][i], 'For Logger params definition necessary'

        if argsTrainer['logger'][i]['type'] == 'comet-ml':
            from pytorch_lightning.loggers.comet import CometLogger
            logger_fn = CometLogger(**argsTrainer['logger'][i]['params'])
        elif argsTrainer['logger'][i]['type'] == 'tensorboard':
            logger_fn = logger.TensorBoardLoggerAdjusted(**argsTrainer['logger'][i]['params'])
        else:
            raise ValueError('Selected logger not implemented!')

        loggers.append(logger_fn)

    return loggers


def execute_training(model: pl.LightningModule, dataLoader, trainer: pl.Trainer) -> None:
    # check if validation_step fct in original LightningModule has been overwritten in model
    is_overwritten = model.validation_step.__code__ is not pl.LightningModule.validation_step.__code__

    if all(getattr(dataLoader, item) is not None for item in ['x_val', 'y_val']) and is_overwritten:
        _logger.debug('Training and validation data included in DataLoader -> Model validation is performed!')
        trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    elif not is_overwritten:
        _logger.debug('Model does not include a validation step -> only Model training is performed')
        trainer.fit(model, train_dataloader=dataLoader.train_dataloader())
    else:
        _logger.warning('NO validation data included in DataLoader but Model has validation step -> Model validation '
                        'is NOT performed (as validation set a tensor filled with zeros is created)!')

        x_empty_size = list(dataLoader.x_train.shape)
        x_empty_size[0] = 1
        y_empty_size = list(dataLoader.y_train.shape)
        y_empty_size[0] = 1
        val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(x_empty_size),
                                                                                    torch.empty(y_empty_size)))

        trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=val_dataloader)


def execute_testing(model: pl.LightningModule, dataLoader, trainer: pl.Trainer) -> None:
    # check if test_step fct in original LightningModule has been overwritten in model
    is_overwritten = model.test_step.__code__ is not pl.LightningModule.test_step.__code__

    if all(getattr(dataLoader, item) is not None for item in ['x_test', 'y_test']) and is_overwritten:
        _logger.debug('test_step included in Model and Test data included in DataLoader -> Model testing performed!')
        trainer.test(model, test_dataloaders=dataLoader.test_dataloader())
    elif is_overwritten:
        _logger.warning('NO test data included in DataLoader but model hast test_step -> testing with zeros tensor!')

        x_empty_size = list(dataLoader.x_train.shape)
        x_empty_size[0] = 1
        y_empty_size = list(dataLoader.y_train.shape)
        y_empty_size[0] = 1
        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(x_empty_size),
                                                                                     torch.empty(y_empty_size)))

        trainer.test(model, test_dataloaders=test_dataloader)
    else:
        _logger.debug('No testing performed since model is missing test_step')
