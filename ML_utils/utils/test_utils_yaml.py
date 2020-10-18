#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.utils.utils_yaml import *
from stfs_pytoolbox.ML_Utils.models import LightningModelTemplate
from stfs_pytoolbox.ML_Utils.loader import DataLoaderTemplate

@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[1]


def test_yaml_structure():
    with pytest.raises(AssertionError):
        general_structure = {'DataLoader': 'params', 'Model': 'params', 'Trainer': 'params'}
        _ = general_structure.pop('DataLoader')
        check_yaml_structure(general_structure)
    with pytest.raises(AssertionError):
        general_structure = {'DataLoader': 'params', 'Model': 'params', 'Trainer': 'params'}
        _ = general_structure.pop('Model')
        check_yaml_structure(general_structure)
    with pytest.raises(AssertionError):
        general_structure = {'DataLoader': 'params', 'Model': 'params', 'Trainer': 'params'}
        _ = general_structure.pop('Trainer')
        check_yaml_structure(general_structure)


def test_check_argsModel():
    yamlTemplate = LightningModelTemplate.yaml_template(key_list=['Model'])
    argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    _ = argsModel.pop('load_model')
    check_argsModel(argsModel)

    # check "type" and "source" flag
    with pytest.raises(AssertionError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        _ = argsModel.pop('type')
        _ = argsModel.pop('load_model')
        check_argsModel(argsModel)
    with pytest.raises(AssertionError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsModel['type'] = 'some other fkt'
        _ = argsModel.pop('load_model')
        check_argsModel(argsModel)
    with pytest.raises(KeyError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        _ = argsModel.pop('load_model')
        _ = argsModel.pop('create_model')
        check_argsModel(argsModel)


def test_check_argsLoader():
    yamlTemplate = DataLoaderTemplate.yaml_template(key_list=['DataLoader'])
    argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    check_argsLoader(argsLoader)

    with pytest.raises(AssertionError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        _ = argsLoader.pop('type')
        check_argsLoader(argsLoader)
    with pytest.raises(AssertionError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsLoader['type'] = 'some other fkt'
        check_argsLoader(argsLoader)


def test_check_argsTrainer():
    yamlTemplate = trainer_yml_template(key_list=['Trainer'])
    argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    check_argsTrainer(argsTrainer)

    with pytest.raises(AssertionError):
        argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        _ = argsTrainer['callbacks'][0].pop('type')
        check_argsTrainer(argsTrainer)
    with pytest.raises(AssertionError):
        argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsTrainer['callbacks'][0]['type'] = 'some other fkt'
        check_argsTrainer(argsTrainer)


def test_replace_keys(path):
    yaml_file = yaml.load(open(path / 'scripts/MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
    yamlTemplate = yaml.load(open(path / 'scripts/SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file = yaml_file.pop('Model001')
    _ = yaml_file.pop('Template')
    yaml_file['DataLoader']['create_DataLoader']['features'] = ['T_0', 'PV']
    yaml_file = replace_keys(yaml_file, yamlTemplate)
    assert yaml_file['DataLoader']['create_DataLoader']['features'] == ['T_0', 'PV'], 'Replacement of keys fails'

    with pytest.raises(AssertionError):  # highest level key not included in the Template
        yaml_file = yaml.load(open(path / 'scripts/MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model001')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['create_model']['n_int'] = 7
        replace_keys(yaml_file, yamlTemplate)
    with pytest.raises(KeyError):  # error in the key_path to the highest level key
        yaml_file = yaml.load(open(path / 'scripts/MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model001')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['create_modle']['n_inp'] = 7
        replace_keys(yaml_file, yamlTemplate)
    with pytest.raises(IndexError):
        yaml_file = yaml.load(open(path / 'scripts/MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file = yaml_file.pop('Model002')
        _ = yaml_file.pop('Template')
        yaml_file['Model']['params']['optimizer']['params']['new_key'] = {'new_key': 7}
        replace_keys(yaml_file, yamlTemplate)