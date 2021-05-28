#######################################################################################################################
# Util functions to execute a ML task (load/ create DataLoader, load/ create Model, start Training)
#######################################################################################################################

# import packages
import argparse
import os
import pytorch_lightning as pl
import torch
from typing import Union, Any

from thunder_torch import _logger  # Logger that are defined in __all__ in the __init__ file
from thunder_torch import callbacks  # Callbacks that are defined in __all__ in the __init__ file
from thunder_torch import _modules_models, _modules_loader, _modules_callbacks, _modules_loss, \
    _modules_optim, _modules_activation, _modules_lr_scheduler
from thunder_torch.utils.general import dynamic_imp, get_ckpt_path


def train_config(argsConfig: dict, argsTrainer: dict) -> dict:
    # check if argsConfig is NoneType (no keys have been given in yaml)
    if argsConfig is None:
        _logger.debug('ArgsConfig dict is empty!')
        return argsTrainer

    # add source path for modules defined in __init__
    if 'source_files' in argsConfig:
        # source_path = os.path.join(os.getcwd(), argsConfig['source_files'] + '.py')
        source_path = argsConfig['source_files']

        if isinstance(source_path, str):
            source_path = [source_path]

        for source_path_run in source_path:
            # Check if module can be imported, exception would be raised within dynamic_imp
            # source path must be full path with .py file extension
            if source_path_run.endswith(".py"):
                source_path_run = source_path_run[:-3]

            if os.path.exists(os.getcwd()+"/"+source_path_run+".py"):
                source_path_run = os.getcwd() + "/" + source_path_run
            elif not os.path.exists(source_path_run + ".py"):
                raise FileNotFoundError(f"Source file for custom function or class does not exists.\nSearched for file: "
                                        f"{source_path_run+'.py'}")
            mod, _ = dynamic_imp(source_path_run)

            if source_path_run in _modules_models:
                _logger.debug(f'Individual Module {source_path_run} already included')
            else:
                _modules_models.append(source_path_run)
                _modules_callbacks.append(source_path_run)
                _modules_optim.append(source_path_run)
                _modules_loss.append(source_path_run)
                _modules_activation.append(source_path_run)
                _modules_lr_scheduler.append(source_path_run)
                _modules_loader.append(source_path_run)
                _logger.debug(f'Individual Module {source_path_run} added')

    # check for deterministic https://pytorch.org/docs/stable/notes/randomness.html
    if 'deterministic' in argsConfig and argsConfig['deterministic'] is True:
        pl.seed_everything(42)
        argsTrainer['params']['deterministic'] = True

    return argsTrainer


def get_model(argsModel: Union[dict, argparse.Namespace]) -> pl.LightningModule:
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
            _, model_cls = dynamic_imp(m, argsModel.type)
            # model_cls = getattr(importlib.import_module(m), argsModel.type)
            _logger.debug(f'Model Class of type {argsModel.type} has been loaded from {m}')
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug(f'Model Class of type {argsModel.type} has NOT been loaded from {m}')

    try:
        if hasattr(argsModel, 'load_model'):
            ckpt_path = get_ckpt_path(argsModel.load_model['path'])
            model = model_cls.load_from_checkpoint(ckpt_path)
            _logger.debug('Model successfully loaded')

        elif hasattr(argsModel, 'create_model'):
            model = model_cls(argparse.Namespace(**argsModel.create_model))
            _logger.debug('Model successfully created')

        else:
            raise KeyError('Model not generated! Either include load_model or create_model dict!')
    except NameError:
        raise NameError(f'Model "{argsModel.type}" cannot be found in given sources: "{_modules_models}"')

    if hasattr(argsModel, 'params'):
        model.hparams_update(update_dict=argsModel.params)
        _logger.debug('model default hparams updated by argsModel.params')
    return model.double()


def get_dataLoader(argsLoader: dict, model: pl.LightningModule = None) -> Any:
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
            _, loader_cls = dynamic_imp(m, argsLoader['type'])
            # loader_cls = getattr(importlib.import_module(m), argsLoader['type'])
            _logger.debug('Model Class of type {} has been loaded from {}'.format(argsLoader['type'], m))
            break
        except AttributeError or ModuleNotFoundError:
            _logger.debug('Model Class of type {} has NOT been loaded from {}'.format(argsLoader['type'], m))
        # assert False, f"{argsLoader['type']} could not be found in {_modules_loader}"

    try:
        if model:
            dataLoader = loader_cls.read_from_yaml(argsLoader, batch=model.hparams.batch,
                                                   num_workers=model.hparams.num_workers)
            model.hparams_update(update_dict={'lparams': dataLoader.lparams})
            _logger.info('DataLoader generated using batch_size and num_workers from model. Loader params are included '
                         'in model.hparams')
        else:
            dataLoader = loader_cls.read_from_yaml(argsLoader)
            _logger.info('DataLoader generated without model information and Loader params not included in model')
    except NameError:
        raise NameError(f'DataLoader "{argsLoader["type"]}" cannot be found in given '
                        f'sources: "{_modules_loader}"')

    return dataLoader


def train_model(model: pl.LightningModule, dataLoader: Any, argsTrainer: dict) -> None:
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

    if 'resume_from_checkpoint' in argsTrainer['params'] and \
            argsTrainer['params']['resume_from_checkpoint'] is not None:
        argsTrainer['params']['resume_from_checkpoint'] = get_ckpt_path(argsTrainer['params']['resume_from_checkpoint'])

    if 'run_epochs' in argsTrainer['params']:
        argsTrainer['params']['max_epochs'] = get_epochs(argsTrainer)

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
        # TODO: in newer versions own keyword for EarlyStopping removed
        if argsTrainer['callbacks'][i]['type'] == 'EarlyStopping':
            earlyStopping = pl.callbacks.EarlyStopping(**argsTrainer['callbacks'][i]['params'])
            argsTrainer['params']['early_stop_callback'] = earlyStopping
        # TODO: in newer version, pre-implemented checkpoint callback changed! better use this one, also the own
        #  keyword is not required anymore
        elif argsTrainer['callbacks'][i]['type'] == 'ModelCheckpoint':
            checkpoint = callbacks.Checkpointing(**argsTrainer['callbacks'][i]['params'])
            argsTrainer['params']['checkpoint_callback'] = checkpoint
        else:
            callback_cls = None
            # Check from which destination the callback class is loaded
            for m in _modules_callbacks:
                try:
                    _, callback_cls = dynamic_imp(m, argsTrainer['callbacks'][i]['type'])
                    # callback_cls = getattr(importlib.import_module(m), argsTrainer['callbacks'][i]['type'])
                    break
                except AttributeError:
                    _logger.debug('Callback of type {} NOT found in {}'.format(argsTrainer['callbacks'][i]['type'], m))
                # assert False, f"{argsTrainer['callbacks'][i]['type']} could not be found in {_modules_callbacks}"

            try:
                if 'params' in argsTrainer['callbacks'][i]:
                    callback = callback_cls(**argsTrainer['callbacks'][i]['params'])
                else:
                    callback = callback_cls()
                callback_list.append(callback)
            except NameError:
                raise NameError(f'Callback "{argsTrainer["callbacks"][i]["type"]}" cannot be found in given '
                                f'sources: "{_modules_callbacks}"')

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
        elif argsTrainer['logger'][i]['type'] == 'TensorBoardLogger':
            logger_fn = pl.loggers.TensorBoardLogger(**argsTrainer['logger'][i]['params'])
        else:
            raise ValueError('Selected logger not implemented!')

        loggers.append(logger_fn)

    return loggers


def get_epochs(argsTrainer: dict) -> int:
    """
    If model loaded from checkpoint and should be trained for a specific number of epochs without knowing at which epoch
    the checkpoint has been saved, the key 'run_epochs' can be used which is added to the ckpt number of epochs and then
    defined as max_epochs for the training. In the case that training is not restored form a checkpoint, run_epochs is
    equal to max_epochs

    Parameters
    ----------
    argsTrainer             - dict with trainer arguments

    """
    if 'resume_from_checkpoint' not in argsTrainer['params']:
        return argsTrainer['params']['run_epochs']

    checkpoint = torch.load(argsTrainer['params']['resume_from_checkpoint'], map_location=lambda storage, loc: storage)
    current_epoch = checkpoint['epoch']
    return argsTrainer['params']['run_epochs'] + current_epoch


def execute_training(model: pl.LightningModule, dataLoader: Any, trainer: pl.Trainer) -> None:
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
        val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(  # type: ignore[attr-defined]
            torch.empty(x_empty_size), torch.empty(y_empty_size)))

        trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=val_dataloader)


def execute_testing(model: pl.LightningModule, dataLoader: Any, trainer: pl.Trainer) -> None:
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
        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(  # type: ignore[attr-defined]
            torch.empty(x_empty_size), torch.empty(y_empty_size)))

        trainer.test(model, test_dataloaders=test_dataloader)
    else:
        _logger.debug('No testing performed since model is missing test_step')
