# Script to add pytest.fixtures with scope=session -> fixtures available in all tests

# import packages
import pandas as pd
import numpy as np
from _pytest.fixtures import FixtureRequest
import pytest
import argparse
import shutil

from thunder_torch.models import *
from thunder_torch.loader import *


# def pytest_configure(config) -> None:
#     plugin = config.pluginmanager.getplugin('mypy')
#     plugin.mypy_argv.append('--check-untyped-defs')


@pytest.fixture(scope='session')
def create_random_df() -> pd.DataFrame:
    example_df = pd.DataFrame(np.random.rand(4, 4))
    example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']
    return example_df


@pytest.fixture(scope='session')
def create_LightningFlexMLP() -> LightningFlexMLP:
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [8]})
    return LightningFlexMLP(hparams)


@pytest.fixture(scope='session')
def create_TabularLoader(create_random_df: pd.DataFrame) -> TabularLoader:
    argsLoader = {'df_samples': create_random_df, 'features': ['T_0', 'P_0'], 'labels': ['yCO2', 'wH2O'],
                  'val_split': {'method': 'percentage', 'params': {'T_0': 0.25}},
                  'test_split': {'method': 'percentage', 'params': {'T_0': 0.25}}}
    return TabularLoader(**argsLoader)


@pytest.fixture(scope="session", autouse=True)
def finalizer(request: FixtureRequest) -> None:
    def clean_directories() -> None:
        shutil.rmtree('checkpoints')
    request.addfinalizer(clean_directories)
