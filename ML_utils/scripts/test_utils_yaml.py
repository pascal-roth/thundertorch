#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.utils.utils_yaml import *


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


def test_yaml_structure(path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    check_yaml_structure(yaml_file)

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('DataLoader')
        check_yaml_structure(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('Model')
        check_yaml_structure(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file.pop('Trainer')
        check_yaml_structure(yaml_file)



@pytest.mark.dependency()
def test_check_argsModel(path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Model')
    check_argsModel(yaml_file)

    # check "type" and "source" flag
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        _ = yaml_file.pop('type')
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['type'] = 'some other fkt'
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        _ = yaml_file.pop('source')
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'some other str'
        check_argsModel(yaml_file)

    # check that necessary parameters are controlled if source=load is selected
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'load'
        _ = yaml_file.pop('load_model')
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'load'
        _ = yaml_file['load_model'].pop('path')
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'create'
        _ = yaml_file.pop('create_model')
        check_argsModel(yaml_file)
    with pytest.raises(ValueError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        yaml_file['source'] = 'some other str'
        _ = yaml_file.pop('create_model')
        check_argsModel(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model')
        _ = yaml_file.pop('create_model')
        _ = yaml_file.pop('load_model')
        check_argsModel(yaml_file)


def test_check_argsLoader(path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    argsLoader = yaml_file.pop('DataLoader')
    check_argsLoader(argsLoader)

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        argsLoader = yaml_file.pop('DataLoader')
        _ = argsLoader.pop('type')
        check_argsLoader(argsLoader)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        argsLoader = yaml_file.pop('DataLoader')
        argsLoader['type'] = 'some other fkt'
        check_argsLoader(argsLoader)


def test_check_argsTrainer(path, create_LightningFlexMLP, create_TabularLoader):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    argsTrainer = yaml_file.pop('Trainer')
    check_argsTrainer(argsTrainer)

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        argsTrainer = yaml_file.pop('Trainer')
        _ = argsTrainer['callbacks'][0].pop('type')
        check_argsTrainer(argsTrainer)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
        argsTrainer = yaml_file.pop('Trainer')
        argsTrainer['callbacks'][0]['type'] = 'some other fkt'
        check_argsTrainer(argsTrainer)


def test_replace_keys(path):
    yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
    yamlTemplate = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Model001')
    _ = yaml_file.pop('Template')
    yaml_file['DataLoader']['create_DataLoader']['features'] = ['T_0', 'PV']
    yaml_file = replace_keys(yaml_file, yamlTemplate)
    assert yaml_file['DataLoader']['create_DataLoader']['features'] == ['T_0', 'PV'], 'Replacement of keys fails'

    with pytest.raises(AssertionError):  # highest level key not included in the Template
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model001')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['create_model']['n_int'] = 7
        replace_keys(yaml_file, yamlTemplate)
    with pytest.raises(KeyError):  # error in the key_path to the highest level key
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model001')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['create_modle']['n_inp'] = 7
        replace_keys(yaml_file, yamlTemplate)
    with pytest.raises(IndexError):
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model002')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['params']['optimizer']['params']['new_key'] = {'new_key': 7}
        replace_keys(yaml_file, yamlTemplate)