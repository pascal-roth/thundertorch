import pytest
import pandas as pd
from pathlib import Path, PosixPath

from thunder_torch.scripts.trainFlexNNmulti import *
from thunder_torch.utils import parse_yaml


@pytest.fixture(scope='module')
def path() -> Path:
    path = Path(__file__).resolve()
    return path.parents[0]


@pytest.mark.dependency()
def test_main(path: Path) -> None:
    with pytest.raises(AssertionError):  # wrong model selected
        yaml_file = parse_yaml(path / 'MultiModelInputEval.yaml')
        yaml_file['config']['model_run'] = ['Model001', 'Model01']
        main(yaml_file)
    with pytest.raises(AssertionError):  # no model selected
        yaml_file = parse_yaml(path / 'MultiModelInputEval.yaml')
        yaml_file['config']['model_run'] = []
        main(yaml_file)
    with pytest.raises(AssertionError):  # wrong config key
        yaml_file = parse_yaml(path / 'MultiModelInputEval.yaml')
        yaml_file['config']['some_key'] = 'some_value'
        main(yaml_file)
    with pytest.raises(AssertionError):  # CPU_per_model and GPU_per_model defined
        yaml_file = parse_yaml(path / 'MultiModelInputEval.yaml')
        yaml_file['config']['cpu_per_model'] = 4
        yaml_file['config']['gpu_per_model'] = 6
        main(yaml_file)


@pytest.mark.dependency(depends=['test_main'])
def test_complete_script(path: Path, create_random_df: pd.DataFrame, tmp_path: PosixPath) -> None:
    yaml_file = parse_yaml(path / 'MultiModelInputEval.yaml', low_key=False)
    yaml_file['Model001']['template'] = path / 'SingleModelInputEval.yaml'
    yaml_file['Model002']['template'] = path / 'SingleModelInputEval.yaml'
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['Model001']['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['Model002']['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['Model001']['Trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    yaml_file['Model002']['Trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    main(yaml_file)
