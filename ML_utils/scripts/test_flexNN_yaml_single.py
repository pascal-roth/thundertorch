#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.flexNN_yaml_single import *
from stfs_pytoolbox.ML_Utils.utils import parse_yaml

@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


# @pytest.mark.dependency(depends=[path() / '.test_training.py'], scope='session')
def test_complete_script(path, create_random_df, tmp_path):
    yaml_file = parse_yaml(path / 'SingleModelInputEval.yaml')
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['dataloader']['create_dataloader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    main(yaml_file)