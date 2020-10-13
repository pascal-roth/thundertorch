#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
from pathlib import Path
import pytest
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.flexNN_yaml_single import *


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


@pytest.mark.dependency()
def test_main(path):
    # test that all three parts of yaml file required
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('DataLoader')
        main(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('Model')
        main(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('Trainer')
        main(yaml_file)


@pytest.mark.dependency()
def test_get_model(path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Model')
    get_model(yaml_file)

    # check "type" and "source" flag
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        _ = yaml_file.pop('type')
        get_model(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        _ = yaml_file.pop('source')
        get_model(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['type'] = 'some other fkt'
        get_model(yaml_file)
    with pytest.raises(ValueError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'some other str'
        get_model(yaml_file)

    # check that necessary parameters are controlled if source=load is selected
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'load'
        _ = yaml_file.pop('load_model')
        get_model(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'load'
        _ = yaml_file['load_model'].pop('path')
        get_model(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'create'
        _ = yaml_file.pop('create_model')
        get_model(yaml_file)


@pytest.mark.dependency()
def test_get_dataloader(path, create_LightningFlexMLP, tmp_path, create_random_df):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
    yaml_file = yaml_file.pop('DataLoader')
    get_dataLoader(yaml_file, create_LightningFlexMLP)

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('DataLoader')
        _ = yaml_file.pop('type')
        get_dataLoader(yaml_file, create_LightningFlexMLP)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('DataLoader')
        yaml_file['type'] = 'some other fkt'
        get_dataLoader(yaml_file, create_LightningFlexMLP)


@pytest.mark.dependency()
def test_train_model(path, create_LightningFlexMLP, create_TabularLoader):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Trainer')
    train_model(create_LightningFlexMLP, create_TabularLoader, yaml_file)

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Trainer')
        _ = yaml_file['callbacks'][0].pop('type')
        train_model(create_LightningFlexMLP, create_TabularLoader, yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Trainer')
        yaml_file['callbacks'][0]['type'] = 'some other fkt'
        train_model(create_LightningFlexMLP, create_TabularLoader, yaml_file)


@pytest.mark.dependency(depends=['test_main', 'test_get_model', 'test_get_dataloader', 'test_train_model'])
def test_complete_script(path, create_random_df, tmp_path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
    main(yaml_file)