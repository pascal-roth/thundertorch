import pytest
import yaml
from pathlib import Path

from stfs_pytoolbox.ML_Utils.flexNN_yaml_multi import *


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


@pytest.mark.dependency()
def test_main(path):
    with pytest.raises(AssertionError):  # wrong model selected
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['Model_run'] = ['Model001', 'Model01']
        main(yaml_file)
    with pytest.raises(AssertionError):  # no model selected
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['Model_run'] = ['Model001', 'Model01']
        main(yaml_file)
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['Nbr_processes'] = 0
        main(yaml_file)


@pytest.mark.dependency()
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


@pytest.mark.dependency(depends=['test_main', 'test_replace_keys'])
def test_complete_script(path, create_random_df, tmp_path):
    yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file['Model001']['Template'] = path / 'SingleModelInputEval.yaml'
    yaml_file['Model002']['Template'] = path / 'SingleModelInputEval.yaml'
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['Model001']['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
    yaml_file['Model002']['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
    main(yaml_file)
