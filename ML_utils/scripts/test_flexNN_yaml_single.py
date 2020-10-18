#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
from pathlib import Path
import pytest

from stfs_pytoolbox.ML_Utils.flexNN_yaml_single import *


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


# @pytest.mark.dependency(depends=[path() / '.test_utils_execute.py'], scope='session')
def test_complete_script(path, create_random_df, tmp_path):
    yaml_file = yaml.load(open(path / 'SingleModelInputEval.yaml'), Loader=yaml.FullLoader)
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    yaml_file['Trainer']['callbacks'][1]['params']['filepath'] = tmp_path / 'model'
    main(yaml_file)