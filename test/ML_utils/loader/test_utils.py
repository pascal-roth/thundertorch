import pytest
import pandas as pd
from typing import Tuple
from pathlib import PosixPath

from thunder_torch.loader._utils import *


@pytest.fixture(scope='module')
def create_random_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    example_df = pd.DataFrame(np.random.rand(4, 4))
    example_df.columns = ['T_0', 'P_0', 'yCO2', 'wH2O']
    x_samples = example_df[['T_0', 'P_0']]
    y_samples = example_df[['yCO2', 'wH2O']]
    return x_samples, y_samples


def test_read_df_from_file(create_random_df: pd.DataFrame, tmp_path: PosixPath) -> None:
    # test format .csv
    create_random_df.to_csv(tmp_path / 'example.csv')
    df_samples = read_df_from_file(tmp_path / 'example.csv')
    assert isinstance(df_samples, pd.DataFrame)

    # test format .h5
    create_random_df.to_hdf(tmp_path / 'example.h5', key='example')
    df_samples = read_df_from_file(tmp_path / 'example.h5')
    assert isinstance(df_samples, pd.DataFrame)

    # test format .txt
    create_random_df.to_csv(tmp_path / 'example.txt')
    df_samples = read_df_from_file(tmp_path / 'example.txt')
    assert isinstance(df_samples, pd.DataFrame)

    # test format .ulf
    #  TODO: implement method if ulf datatype is know

    # test format .csv with different separator
    create_random_df.to_csv(tmp_path / 'example.csv', sep=':')
    df_samples = read_df_from_file(tmp_path / 'example.csv', sep=':')
    assert isinstance(df_samples, pd.DataFrame)

    # Error if delimiter is wrong
    with pytest.raises(AssertionError):
        create_random_df.to_csv(tmp_path / 'example.csv', sep=':')
        read_df_from_file(tmp_path / 'example.csv')

    # Error for other file type
    with pytest.raises(TypeError):
        read_df_from_file('some_file.other_type')


def test_split_data_random(create_random_dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    x_samples, y_samples = create_random_dataset

    with pytest.raises(AssertionError):
        # Percentage defined in split_params cannot exceed 1.0
        data_split_random(x_samples=x_samples, y_samples=y_samples, split_params=1.2)
    with pytest.raises(AssertionError):
        # Percentage defined in split_params has to be of type int
        # arg type error ignored since assertion error if wrong type is given is controlled
        data_split_random(x_samples=x_samples, y_samples=y_samples, split_params='str')  # type: ignore[arg-type]

    x_samples, x_split, y_samples, y_split = data_split_random(x_samples=x_samples, y_samples=y_samples,
                                                               split_params=0.5)
    assert x_samples.shape == (2, 2), 'Split failed'
    assert x_split.shape == (2, 2), 'Split failed'


def test_split_data_percentage(create_random_dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    x_samples, y_samples = create_random_dataset

    with pytest.raises(AssertionError):  # percentage too low
        data_split_percentage(x_samples=x_samples, y_samples=y_samples, split_params={'T_0': 0.05})
    with pytest.raises(AssertionError):  # percentage too large
        data_split_percentage(x_samples=x_samples, y_samples=y_samples, split_params={'T_0': 1.2})
    with pytest.raises(AssertionError):  # feature missing
        # arg type error ignored since assertion error if wrong type is given is controlled
        data_split_percentage(x_samples=x_samples, y_samples=y_samples, split_params=0.5)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):  # feature not in data
        data_split_percentage(x_samples=x_samples, y_samples=y_samples, split_params={'T_1': 0.25})

    x_samples, x_split, y_samples, y_split = data_split_percentage\
        (x_samples=x_samples, y_samples=y_samples, split_params={'T_0': 0.5})

    assert isinstance(x_samples, pd.DataFrame) and not x_samples.empty, 'Returned x_samples not correct'
    assert isinstance(x_split, pd.DataFrame) and not x_split.empty, 'Returned x_split not correct'
    assert isinstance(y_samples, pd.DataFrame) and not y_samples.empty, 'Returned y_samples not correct'
    assert isinstance(y_split, pd.DataFrame) and not y_split.empty, 'Returned y_split not correct'

    assert x_samples.shape == (2, 2), 'Split not performed correct'
    assert x_split.shape == (2, 2), 'Split not performed correct'


def test_split_data_explicit(create_random_dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    x_samples, y_samples = create_random_dataset

    with pytest.raises(TypeError):  # wrong feature value type
        data_split_explicit(x_samples=x_samples, y_samples=y_samples, split_params={'T_0': 'false type'})
    with pytest.raises(AssertionError):  # wrong feature value
        data_split_explicit(x_samples=x_samples, y_samples=y_samples, split_params={'T_0': 5})
    with pytest.raises(AssertionError):  # feature missing
        # arg type error ignored since assertion error if wrong type is given is controlled
        data_split_explicit(x_samples=x_samples, y_samples=y_samples, split_params=0.5)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):  # feature not in data
        data_split_explicit(x_samples=x_samples, y_samples=y_samples, split_params={'T_1': x_samples['T_0'].iloc[1]})

    x_samples, x_split, y_samples, y_split = data_split_explicit\
        (x_samples=x_samples, y_samples=y_samples, split_params={'T_0': x_samples['T_0'].iloc[1]})

    assert isinstance(x_samples, pd.DataFrame) and not x_samples.empty, 'Returned x_samples not correct'
    assert isinstance(x_split, pd.DataFrame) and not x_split.empty, 'Returned x_split not correct'
    assert isinstance(y_samples, pd.DataFrame) and not y_samples.empty, 'Returned y_samples not correct'
    assert isinstance(y_split, pd.DataFrame) and not y_split.empty, 'Returned y_split not correct'

    assert x_samples.shape == (3, 2), 'Split not performed correct'
    assert x_split.shape == (1, 2), 'Split not performed correct'
