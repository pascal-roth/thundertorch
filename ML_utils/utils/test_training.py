#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.utils.training import *
from stfs_pytoolbox.ML_Utils.loader import TabularLoader
from stfs_pytoolbox.ML_Utils.utils import parse_yaml

@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


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


def test_train_model(path, create_LightningFlexMLP, create_TabularLoader):
    yaml_file = parse_yaml(path / 'MinimalSingleModelInputEval.yml')
    argsTrainer = yaml_file.pop('trainer')
    train_model(create_LightningFlexMLP, create_TabularLoader, argsTrainer)
