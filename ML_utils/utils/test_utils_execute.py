#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
from pathlib import Path
import pytest

import torch
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.utils.utils_execute import *
from stfs_pytoolbox.ML_Utils.loader import TabularLoader


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


def test_get_model(path):
    yaml_file = yaml.load(open(path / 'MinimalSingleModelInputEval.yml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Model')
    model = get_model(yaml_file)
    assert isinstance(model, pl.LightningModule)
    assert model.dtype == torch.float64


def test_get_dataloader(path, create_LightningFlexMLP, tmp_path, create_random_df):
    yaml_file = yaml.load(open(path / 'MinimalSingleModelInputEval.yml'), Loader=yaml.FullLoader)
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    argsLoader = yaml_file.pop('DataLoader')
    dataLoader = get_dataLoader(argsLoader, create_LightningFlexMLP)
    assert isinstance(dataLoader, TabularLoader)


def test_train_model(path, create_LightningFlexMLP, create_TabularLoader, tmp_path):
    yaml_file = yaml.load(open(path / 'MinimalSingleModelInputEval.yml'), Loader=yaml.FullLoader)
    argsTrainer = yaml_file.pop('Trainer')
    train_model(create_LightningFlexMLP, create_TabularLoader, argsTrainer)
