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
        yaml_file['config']['model_run'] = ['Model001', 'Model01']
        main(yaml_file)
    with pytest.raises(AssertionError):  # no model selected
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['config']['model_run'] = []
        main(yaml_file)
    with pytest.raises(AssertionError):  # wrong config key
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['config']['some_key'] = 'some_value'
        main(yaml_file)
    with pytest.raises(AssertionError):  # CPU_per_model and GPU_per_model defined
        yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['config']['CPU_per_model'] = 4
        yaml_file['config']['GPU_per_model'] = 6
        main(yaml_file)


@pytest.mark.dependency(depends=['test_main'])
def test_complete_script(path, create_random_df, tmp_path):
    yaml_file = yaml.load(open(path / 'MultiModelInputEval.yaml'), Loader=yaml.FullLoader)
    yaml_file['Model001']['Template'] = path / 'SingleModelInputEval.yaml'
    yaml_file['Model002']['Template'] = path / 'SingleModelInputEval.yaml'
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['Model001']['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['Model002']['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['Model001']['Trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    yaml_file['Model002']['Trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    main(yaml_file)
