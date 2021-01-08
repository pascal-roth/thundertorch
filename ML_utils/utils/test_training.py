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
from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP
from stfs_pytoolbox.ML_Utils.utils import parse_yaml


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


def test_config_source_files(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        argsConfig = {'source_files': str(tmp_path)}
        train_config(argsConfig, argsTrainer={})


def test_config_reproducibility(create_TabularLoader):
    argsConfig = {'reproducibility': True}
    argsTrainer = {'params': {'max_epochs': 2, 'logger': False}}
    argsTrainer = train_config(argsConfig, argsTrainer)

    print(argsTrainer)
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [8]})
    model_1 = LightningFlexMLP(hparams)

    trainer_1 = pl.Trainer(**argsTrainer['params'])
    trainer_1.fit(model_1, train_dataloader=create_TabularLoader.train_dataloader(),
                  val_dataloaders=create_TabularLoader.val_dataloader())
    trainer_2 = pl.Trainer(**argsTrainer['params'])
    trainer_2.fit(model_1, train_dataloader=create_TabularLoader.train_dataloader(),
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
def test_callbacks():
    pass


@pytest.mark.dependency()
def test_logger():
    pass


@pytest.mark.dependency(depend=['test_callbacks', 'test_logger'])
def test_train_model(path, create_LightningFlexMLP, create_TabularLoader):
    yaml_file = parse_yaml(path / 'MinimalSingleModelInputEval.yml')
    argsTrainer = yaml_file.pop('trainer')
    train_model(create_LightningFlexMLP, create_TabularLoader, argsTrainer)
