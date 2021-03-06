#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
from pathlib import Path
import pytest
import os

from thunder_torch.utils.yaml import *
from thunder_torch.models import LightningModelTemplate
from thunder_torch.loader import DataLoaderTemplate


@pytest.fixture(scope='module')
def path() -> Path:
    path = Path(__file__).resolve()
    return path.parents[1]


@pytest.mark.dependency()
def test_lower_keys() -> None:
    example_dict_v0 = {'DataLoader': 'params', 'model': 'params', 'Trainer': 'params'}
    example_dict_v0 = lower_keys(example_dict_v0)
    assert example_dict_v0['dataloader'] == 'params', 'does not convert to lower keys'
    assert example_dict_v0['trainer'] == 'params', 'does not convert to lower keys'

    example_dict_v1: dict = {'DataLoader': 'params', 'model': {'n_inp': 3, 'N_out': 4}, 'trainer': 'params'}
    example_dict_v1 = lower_keys(example_dict_v1)
    assert example_dict_v1['model']['n_out'] == 4, 'Recursion fails'

    example_dict_v3: dict = {'DataLoader': 'params', 'model': 'params', 'Trainer': [{'callback_1': 'param'},
                                                                                    {'CallBack_2': 3}]}
    example_dict_v3 = lower_keys(example_dict_v3)
    assert example_dict_v3['trainer'][1]['callback_2'] == 3, 'List access fails'

    example_dict_v4: dict = {'DataLoader': {'create_DataLoader': {'split_data': {'method': 'explicit', 'T_0': 745}}},
                             'model': 'params', 'trainer': 'params'}
    example_dict_v4 = lower_keys(example_dict_v4)
    assert example_dict_v4['dataloader']['create_dataloader']['split_data']['T_0'] == 745, 'Split_data exception fails'


def test_yaml_structure() -> None:
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


def test_check_argsModel() -> None:
    yamlTemplate = LightningModelTemplate.yaml_template(key_list=['Model'])
    argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    argsModel = lower_keys(argsModel)
    _ = argsModel.pop('load_model')
    _ = argsModel.pop('###info###')
    check_argsModel(argsModel)

    # check OptionClass Object
    with pytest.raises(AssertionError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsModel = lower_keys(argsModel)
        _ = argsModel.pop('type')
        _ = argsModel.pop('load_model')
        _ = argsModel.pop('###info###')
        check_argsModel(argsModel)
    with pytest.raises(AttributeError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsModel['type'] = 'some other fkt'
        argsModel = lower_keys(argsModel)
        _ = argsModel.pop('load_model')
        _ = argsModel.pop('###info###')
        check_argsModel(argsModel)

    # check model source error
    with pytest.raises(KeyError):
        argsModel = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsModel = lower_keys(argsModel)
        _ = argsModel.pop('load_model')
        _ = argsModel.pop('create_model')
        _ = argsModel.pop('###info###')
        check_argsModel(argsModel)


def test_check_argsLoader() -> None:
    yamlTemplate = DataLoaderTemplate.yaml_template(key_list=['DataLoader'])
    argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    argsLoader = lower_keys(argsLoader)
    _ = argsLoader.pop('load_dataloader')
    _ = argsLoader.pop('###info###')
    check_argsLoader(argsLoader)

    # check OptionClass Object
    with pytest.raises(AssertionError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsLoader = lower_keys(argsLoader)
        _ = argsLoader.pop('type')
        _ = argsLoader.pop('load_dataloader')
        _ = argsLoader.pop('###info###')
        check_argsLoader(argsLoader)
    with pytest.raises(AttributeError):
        argsLoader = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsLoader = lower_keys(argsLoader)
        argsLoader['type'] = 'some other fkt'
        _ = argsLoader.pop('load_dataloader')
        _ = argsLoader.pop('###info###')
        check_argsLoader(argsLoader)


def test_check_argsTrainer() -> None:
    yamlTemplate = trainer_yml_template(key_list=['Trainer'])
    argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
    _ = argsTrainer['params'].pop('gpus')
    check_argsTrainer(argsTrainer)

    with pytest.raises(AssertionError):
        argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        _ = argsTrainer['callbacks'][0].pop('type')
        check_argsTrainer(argsTrainer)
    with pytest.raises(AttributeError):
        argsTrainer = yaml.load(yamlTemplate, Loader=yaml.FullLoader)
        argsTrainer['callbacks'][0]['type'] = 'some other fkt'
        check_argsTrainer(argsTrainer)


@pytest.mark.dependency(depends=['test_lower_keys'])
def test_get_argsModels(path: Path) -> None:
    # without config tree
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    _ = argsMulti.pop('config')
    argsModels, argsConfig = get_argsModel(argsMulti)
    assert argsModels == argsMulti, 'filter accesses model_run, which should not be included'
    assert argsConfig == [], 'argsConfig are created at some point, normally should not be present'

    # with config tree
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsMulti['config']['model_run'] = 'Model001'
    argsModels, argsConfig = get_argsModel(argsMulti)
    assert argsModels == {'Model001': argsMulti['Model001']}, 'model_run filter not working'
    assert 'config' not in argsMulti, 'argsConfig not separated correctly from argsMulti'
    assert 'config' not in argsModels, 'Config file not removed from argsModels'

    with pytest.raises(AssertionError):  # model name included in model_run not found
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        argsMulti['config']['model_run'] = 'model1'
        get_argsModel(argsMulti)
    with pytest.raises(AssertionError):  # model_run is empty
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        argsMulti['config']['model_run'] = []
        get_argsModel(argsMulti)

    # with empty config tree
    with pytest.raises(AssertionError):
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        _ = argsMulti['config'].pop('model_run')
        get_argsModel(argsMulti)


@pytest.mark.dependency(depends=['test_lower_keys'])
def test_replace_keys(path: Path) -> None:
    yaml_file = parse_yaml(path / 'scripts/MultiModelInputEval.yaml')
    yamlTemplate = parse_yaml(path / 'scripts/SingleModelInputEval.yaml')

    # check general ability to replace keys
    yaml_file_001 = yaml_file.pop('model001')
    _ = yaml_file_001.pop('template')
    yaml_file_001['dataloader']['create_dataloader']['features'] = ['T_0', 'PV']
    yaml_file_001 = replace_keys(yaml_file_001, yamlTemplate)
    assert yaml_file_001['dataloader']['create_dataloader']['features'] == ['T_0', 'PV'], 'Replacement of keys fails'

    # check ability to work with "idx" and "type" key when lists are manipulated
    yaml_file_002 = yaml_file.pop('model002')
    _ = yaml_file_002.pop('template')
    yaml_file_002 = replace_keys(yaml_file_002, yamlTemplate)
    assert yaml_file_002['trainer']['callbacks'][0]['params']['patience'] == 15, \
        'Manipulation of list entry with "type" key fails'
    assert yaml_file_002['trainer']['callbacks'][1]['params']['save_top_k'] == 3, \
        'Manipulation of list entry with "idx" key fails'

    with pytest.raises(KeyError):  # error in the key_path to the highest level key
        yaml_file = parse_yaml(path / 'scripts/MultiModelInputEval.yaml')
        yaml_file = yaml_file.pop('model001')
        _ = yaml_file.pop('template')
        yaml_file['model']['create_modle']['n_inp'] = 7
        replace_keys(yaml_file, yamlTemplate)


def test_replace_expression() -> None:
    example_dict_v0: dict = {'dataloader': 'params', 'model': 'params', 'trainer': 'params_<model_name>'}
    example_dict_v0 = replace_expression(example_dict_v0, 'Model001')
    assert example_dict_v0['trainer'] == 'params_Model001', 'does not find expression'

    example_dict_v1: dict = {'trainer': {'max_epochs': 3, 'load_from_checkpoint': './checkpoints/<model_name>'}}
    example_dict_v1 = replace_expression(example_dict_v1, 'Model001')
    assert example_dict_v1['trainer']['load_from_checkpoint'] == './checkpoints/Model001', 'Recursion fails'

    example_dict_v2: dict = {'trainer': {'max_epochs': 3, 'load_from_checkpoint': './checkpoints/{model_name_new}'}}
    example_dict_v2 = replace_expression(example_dict_v2, 'Model001', '{model_name_new}')
    assert example_dict_v2['trainer']['load_from_checkpoint'] == './checkpoints/Model001', 'Expression adjustment fails'

    example_dict_v3: dict = {'DataLoader': 'params', 'model': 'params', 'trainer':
        [{'callback_1': 'param'}, {'callback_2': {'load_from_checkpoint': './checkpoints/<model_name>'}}]}
    example_dict_v3 = replace_expression(example_dict_v3, 'Model001')
    assert example_dict_v3['trainer'][1]['callback_2']['load_from_checkpoint'] == './checkpoints/Model001', 'List fails'

    example_dict_v4: dict = {'model': ['label_1', '<model_name>', 'label_3']}
    example_dict_v4 = replace_expression(example_dict_v4, 'Model001')
    assert example_dict_v4['model'][1] == 'Model001', 'List of str fails'


@pytest.mark.dependency(depends=['test_replace_keys', 'test_get_argsModels'])
def test_get_argsDict(path: Path) -> None:
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsMulti['Model001']['template'] = argsMulti['Model002']['template'] = str(
        path / 'scripts/SingleModelInputEval.yaml')
    argsMulti['Model001']['Trainer']['callbacks'][1]['params']['filepath'] = '<model_name>'
    argsModels, argsConfig = get_argsModel(argsMulti)
    model_dicts = get_argsDict(argsModels)
    assert len(model_dicts) == 2, 'dict appending fails'
    assert 'template' not in model_dicts[0], 'removal of template key fails'
    assert model_dicts[0]['trainer']['callbacks'][1]['params']['filepath'] == 'Model001', \
        'replace expression with capital letter model name fails'

    with pytest.raises(AssertionError):  # template definition missing
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        _ = argsMulti['Model001'].pop('Template')
        argsModels, argsConfig = get_argsModel(argsMulti)
        get_argsDict(argsModels)

@pytest.mark.dependency(depends=['test_get_argsModels'])
def test_get_num_processes(path: Path) -> None:
    # without definition of cpu_per_model and gpu_per_model
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsMulti['Model001']['template'] = argsMulti['Model002']['template'] = str(
        path / 'scripts/SingleModelInputEval.yaml')
    argsModels, argsConfig = get_argsModel(argsMulti)
    nbr_processes, list_gpu = get_num_processes(argsConfig, argsModels)
    assert nbr_processes == 1, 'default nbr of processes for the given case not changed at some point'
    assert list_gpu == [0], 'default list if no gpu is given is wrong'

    # with definition of gpu_per_model
    with pytest.raises(AssertionError):
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        argsModels, argsConfig = get_argsModel(argsMulti)
        argsConfig['gpu_per_model'] = 1
        get_num_processes(argsConfig, argsModels)

    # with defintion of cpu_per_model  # TODO: changed for docker which can hurt the sense of the test, control that
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsModels, argsConfig = get_argsModel(argsMulti)
    # argsConfig['cpu_per_model'] = max(int(os.cpu_count() / 2), 1)
    argsConfig['cpu_per_model'] = 1
    nbr_processes, list_gpu = get_num_processes(argsConfig, argsModels)
    assert nbr_processes == 1, f'nbr_processes "{nbr_processes}" intended to be 1'

    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsModels, argsConfig = get_argsModel(argsMulti)
    # argsConfig['cpu_per_model'] = max(int(os.cpu_count() / 4), 1)
    argsConfig['cpu_per_model'] = 1
    nbr_processes, list_gpu = get_num_processes(argsConfig, argsModels)
    assert nbr_processes == 1, f'nbr_processes "{nbr_processes}" has to be downgraded to 2 since only two models are ' \
                               f'available'

    with pytest.raises(AssertionError):  # nbr of cpu_per_model exceeds available cpu's
        argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
        argsModels, argsConfig = get_argsModel(argsMulti)
        argsConfig['cpu_per_model'] = os.cpu_count() + 1  #type: ignore[operator]
        get_num_processes(argsConfig, argsModels)

    # with definition of nbr_processes
    argsMulti = parse_yaml(path / 'scripts/MultiModelInputEval.yaml', low_key=False)
    argsModels, argsConfig = get_argsModel(argsMulti)
    # argsConfig['cpu_per_model'] = max(int(os.cpu_count() / 2), 1)
    argsConfig['cpu_per_model'] = 1
    argsConfig['nbr_processes'] = 1
    nbr_processes, list_gpu = get_num_processes(argsConfig, argsModels)
    assert nbr_processes == 1, f'nbr_processes "{nbr_processes}" has to be downgraded to 1 since the defined nbr of ' \
                               f'cannot be exceeded'
