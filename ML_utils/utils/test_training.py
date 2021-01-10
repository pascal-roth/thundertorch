#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
from pathlib import Path
import pytest
import argparse
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.utils.training import *
from stfs_pytoolbox.ML_Utils.loader import TabularLoader
from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP, LightningModelBase
from stfs_pytoolbox.ML_Utils.utils import parse_yaml
from stfs_pytoolbox.ML_Utils import callbacks
from stfs_pytoolbox.ML_Utils import logger

from .MinimalLightningModel import MinimalLightningModule


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


def test_config_source_files(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        argsConfig = {'source_files': str(tmp_path)}
        train_config(argsConfig, argsTrainer={})


def test_config_reproducibility(create_TabularLoader, create_LightningFlexMLP):
    argsConfig = {'reproducibility': True}
    argsTrainer = {'params': {'max_epochs': 2, 'logger': False}}
    argsTrainer = train_config(argsConfig, argsTrainer)

    trainer_1 = pl.Trainer(**argsTrainer['params'])
    trainer_1.fit(create_LightningFlexMLP, train_dataloader=create_TabularLoader.train_dataloader(),
                  val_dataloaders=create_TabularLoader.val_dataloader())
    trainer_2 = pl.Trainer(**argsTrainer['params'])
    trainer_2.fit(create_LightningFlexMLP, train_dataloader=create_TabularLoader.train_dataloader(),
                  val_dataloaders=create_TabularLoader.val_dataloader())

    print('Model 1', trainer_1.tng_tqdm_dic['loss'])
    print('Model 2', trainer_2.tng_tqdm_dic['loss'])

    assert trainer_1.tng_tqdm_dic['loss'] == trainer_2.tng_tqdm_dic['loss'], 'reproducibility failed'


def test_get_model(path):
    yaml_file = parse_yaml(path / 'MinimalSingleModelInputEval.yml')
    yaml_file = yaml_file.pop('model')
    model = get_model(yaml_file)
    assert isinstance(model, pl.LightningModule)
    assert model.dtype == torch.float64


def test_get_dataloader(path, create_LightningFlexMLP, tmp_path, create_random_df):
    yaml_file = parse_yaml(path / 'MinimalSingleModelInputEval.yml')
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['dataloader']['create_dataloader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    argsLoader = yaml_file.pop('dataloader')
    dataLoader = get_dataLoader(argsLoader, create_LightningFlexMLP)
    assert isinstance(dataLoader, TabularLoader)


@pytest.mark.dependency()
def test_train_callbacks():
    # check handling of EarlyStopping and Checkpointing callback since they have their own trainer flags
    argsTrainer = {'callbacks': [{'type': 'EarlyStopping',
                                  'params': {'monitor': 'val_loss', 'patience': 12, 'mode': 'min'}},
                                 {'type': 'Checkpointing',
                                  'params': {'filepath': 'checkpoints/some_name', 'save_top_k': 1, 'period': 0}}],
                   'params': {'max_epochs': 3}}
    argsTrainer = train_callbacks(argsTrainer)

    assert isinstance(argsTrainer['params']['early_stop_callback'], pl.callbacks.EarlyStopping), \
        'EarlyStopping Handling fails'
    assert argsTrainer['params']['early_stop_callback'].patience == 12, 'EarlyStopping params Handling fail'

    assert isinstance(argsTrainer['params']['checkpoint_callback'], callbacks.Checkpointing), \
        'Checkpointing Handling fails'
    assert argsTrainer['params']['checkpoint_callback'].filename == 'some_name', 'Checkpointing params Handling fail'

    assert argsTrainer['params']['callbacks'] == [], 'Removal of the arg dicts fails'

    # check handling of other callbacks
    argsTrainer = {'callbacks': [{'type': 'Explained_Variance'},
                                 {'type': 'ProgressBar',
                                  'params': {'refresh_rate': 3}}],
                   'params': {'max_epochs': 3}}
    argsTrainer = train_callbacks(argsTrainer)

    assert isinstance(argsTrainer['params']['callbacks'][0], callbacks.Explained_Variance), \
        'Callback initialization from Toolbox fails'
    assert isinstance(argsTrainer['params']['callbacks'][1], pl.callbacks.ProgressBar), \
        'Callback initialization from Lightning fails'
    assert argsTrainer['params']['callbacks'][1].refresh_rate == 3, 'Param Handling fails'

    assert 'checkpoint_callback' not in argsTrainer['params'], 'Checkpointing callback initialized but not intended'
    assert 'early_stop_callback' not in argsTrainer['params'], 'EarlyStopping callback initialized but not intended'


@pytest.mark.dependency()
def test_train_logger():  # control of comet logger fails even if it is working 
    # argsTrainer = {'params': {'max_epochs': 3},
    #                'logger': [{'type': 'comet-ml',
    #                            'params': {'api_key': 'ENlkidpOntcgkoGGs5nkyhFv5', 'project_name': 'general',
    #                                       'workspace': 'proth', 'experiment_name': 'try_out'}},
    #                           {'type': 'tensorboard',
    #                            'params': {'save_dir': 'logs/'}}]}
    argsTrainer = {'params': {'max_epochs': 3},
                   'logger': [{'type': 'tensorboard',
                               'params': {'save_dir': 'logs/'}}]}
    argsTrainer['params']['logger'] = train_logger(argsTrainer)

    # assert isinstance(argsTrainer['params']['logger'][0], pl.loggers.comet.CometLogger), 'Comit init fails'
    assert isinstance(argsTrainer['params']['logger'][0], logger.TensorBoardLoggerAdjusted), 'Tensorboard init fails'

    with pytest.raises(AssertionError):  # Logger misses params
        argsTrainer = {'params': {'max_epochs': 3},
                       'logger': [{'type': 'tensorboard'}]}
        train_logger(argsTrainer)

    with pytest.raises(ValueError):  # not implemented logger
        argsTrainer = {'params': {'max_epochs': 3},
                       'logger': [{'type': 'some_logger',
                                   'params': {'some_param': 'some_value'}}]}
        train_logger(argsTrainer)


@pytest.mark.dependency()
def test_execute_training(create_TabularLoader, create_LightningFlexMLP, create_random_df):
    # modal has val_step and dataloader includes a validation set
    trainer_a = pl.Trainer(fast_dev_run=True)
    execute_training(create_LightningFlexMLP, create_TabularLoader, trainer_a)
    assert trainer_a.tng_tqdm_dic['val_loss'] != 0, 'validation not performed'

    # model does not has val_step
    model_red_val = MinimalLightningModule(argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [8]}))
    trainer_b = pl.Trainer(fast_dev_run=True)
    execute_training(model_red_val, create_TabularLoader, trainer_b)
    assert 'val_loss' not in trainer_b.tng_tqdm_dic, 'validation_step independence fails'

    # model has val_step, but dataloader does not a validation set
    argsLoader = {'df_samples': create_random_df, 'features': ['T_0', 'P_0'], 'labels': ['yCO2', 'wH2O']}
    dataLoader = TabularLoader(**argsLoader)
    trainer_c = pl.Trainer(fast_dev_run=True)
    execute_training(create_LightningFlexMLP, dataLoader, trainer_c)
    assert trainer_c.tng_tqdm_dic['val_loss'] != 0, 'validation not performed'


@pytest.mark.dependency()
def test_execute_testing(create_TabularLoader, create_LightningFlexMLP, create_random_df):
    # modal has test_step and dataloader includes a test set
    trainer_a = pl.Trainer(fast_dev_run=True, callbacks=[])
    execute_testing(create_LightningFlexMLP, create_TabularLoader, trainer_a)
    assert 'test_loss' in trainer_a.callback_metrics, 'test not performed'

    # model does not has test_step
    model_red_val = MinimalLightningModule(argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [8]}))
    trainer_b = pl.Trainer(fast_dev_run=True)
    execute_testing(model_red_val, create_TabularLoader, trainer_b)
    assert 'test_loss' not in trainer_b.callback_metrics, 'test_step independence fails'

    # model has test_step, but dataloader does not have a test set
    argsLoader = {'df_samples': create_random_df, 'features': ['T_0', 'P_0'], 'labels': ['yCO2', 'wH2O']}
    dataLoader = TabularLoader(**argsLoader)
    trainer_c = pl.Trainer(fast_dev_run=True)
    execute_testing(create_LightningFlexMLP, dataLoader, trainer_c)
    assert 'test_loss' in trainer_c.callback_metrics, 'test not performed'


@pytest.mark.dependency(depends=['test_train_callbacks', 'test_train_logger', 'test_execute_training',
                                 'test_execute_testing'])
def test_train_model(path, create_LightningFlexMLP, create_TabularLoader):
    yaml_file = parse_yaml(path / 'MinimalSingleModelInputEval.yml')
    argsTrainer = yaml_file.pop('trainer')
    train_model(create_LightningFlexMLP, create_TabularLoader, argsTrainer)
