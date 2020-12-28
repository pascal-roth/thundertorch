#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.utils.yaml import *
from stfs_pytoolbox.ML_Utils.models import LightningModelTemplate
from stfs_pytoolbox.ML_Utils.loader import DataLoaderTemplate
from stfs_pytoolbox.ML_Utils.utils import parse_yaml

@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[1]


def test_yaml_structure():
    with pytest.raises(AssertionError):
        general_structure = {'dataloader': 'params', 'model': 'params', 'trainer': 'params'}
        _ = general_structure.pop('dataloader')
        check_yaml_structure(general_structure)
    with pytest.raises(AssertionError):
        general_structure = {'dataloader': 'params', 'model': 'params', 'trainer': 'params'}
        _ = general_structure.pop('model')
        check_yaml_structure(general_structure)
    with pytest.raises(AssertionError):
        general_structure = {'dataloader': 'params', 'model': 'params', 'trainer': 'params'}
        _ = general_structure.pop('trainer')
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
    argsLoader = lower_keys(argsLoader)
    _ = argsLoader.pop('load_dataloader')
    _ = argsLoader.pop('###info###')
    check_argsLoader(argsLoader)

    with pytest.raises(AssertionError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsLoader = lower_keys(argsLoader)
        _ = argsLoader.pop('type')
        _ = argsLoader.pop('load_dataloader')
        _ = argsLoader.pop('###info###')
        check_argsLoader(argsLoader)
    with pytest.raises(AssertionError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsLoader = lower_keys(argsLoader)
        argsLoader['type'] = 'some other fkt'
        _ = argsLoader.pop('load_dataloader')
        _ = argsLoader.pop('###info###')
        check_argsLoader(argsLoader)


def test_check_argsTrainer():
    yamlTemplate = trainer_yml_template(key_list=['Trainer'])
    argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    _ = argsTrainer['params'].pop('gpus')
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
    yaml_file = parse_yaml(path / 'scripts/MultiModelInputEval.yaml')
    yamlTemplate = parse_yaml(path / 'scripts/SingleModelInputEval.yaml')
    yaml_file = yaml_file.pop('model001')
    _ = yaml_file.pop('template')
    yaml_file['dataloader']['create_dataloader']['features'] = ['T_0', 'PV']
    yaml_file = replace_keys(yaml_file, yamlTemplate)
    assert yaml_file['dataloader']['create_dataloader']['features'] == ['T_0', 'PV'], 'Replacement of keys fails'

    with pytest.raises(KeyError):  # error in the key_path to the highest level key
        yaml_file = parse_yaml(path / 'scripts/MultiModelInputEval.yaml')
        yaml_file = yaml_file.pop('model001')
        _ = yaml_file.pop('template')
        yaml_file['model']['create_modle']['n_inp'] = 7
        replace_keys(yaml_file, yamlTemplate)
