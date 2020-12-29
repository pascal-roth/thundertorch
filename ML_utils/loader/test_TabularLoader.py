import pytest
import torch
import yaml
import argparse
from pathlib import Path
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.loader import TabularLoader
from stfs_pytoolbox.ML_Utils.callbacks import Checkpointing
from stfs_pytoolbox.ML_Utils.utils import parse_yaml


@pytest.fixture(scope='module')
def create_example_TabularLoader(create_random_df):
    return TabularLoader(create_random_df, features=['T_0', 'P_0'], labels=['yCO2', 'wH2O'])


@pytest.fixture(scope='module')
def path():
    path = Path(__file__).resolve()
    return path.parents[0]


@pytest.mark.dependency()
def test_init(create_random_df):
    example_df = create_random_df

    Loader = TabularLoader(example_df, features=['T_0', 'P_0'], labels=['yCO2', 'wH2O'])
    assert isinstance(Loader, TabularLoader), 'Returned object does not match class TabularLoader'

    with pytest.raises(AssertionError):
        TabularLoader(example_df, features=['T_0', 69], labels=['yCO2', 'wH2O'])
    with pytest.raises(AssertionError):
        TabularLoader(example_df, features=['T_0', 'P_0'], labels=['yCO2', 69])
    with pytest.raises(AssertionError):
        TabularLoader(example_df, features=['T_0', 'P_0'], labels=['T_0', 'wH2O'])
    with pytest.raises(KeyError):
        TabularLoader(example_df, features=['T_0', 'some other str'], labels=['yCO2', 'wH2O'])


@pytest.mark.dependency(depends=['test_init'])
def test_add_val_data(create_random_df, create_example_TabularLoader, tmp_path):
    create_random_df.to_csv(tmp_path / 'example.csv')
    create_example_TabularLoader.add_val_data(tmp_path / 'example.csv')


@pytest.mark.dependency(depends=['test_init'])
def test_val_split(create_example_TabularLoader):
    Loader = create_example_TabularLoader
    with pytest.raises(AttributeError):
        Loader.val_split(method='some other str')
    with pytest.raises(AssertionError):
        Loader.val_split(method='explicit')
    with pytest.raises(AssertionError):
        Loader.val_split(method='percentage')


@pytest.mark.dependency(depends=['test_init'])
def test_add_test_data(create_random_df, create_example_TabularLoader, tmp_path):
    create_random_df.to_csv(tmp_path / 'example.csv')
    create_example_TabularLoader.add_test_data(tmp_path / 'example.csv')


@pytest.mark.dependency(depends=['test_init'])
def test_test_split(create_example_TabularLoader):
    Loader = create_example_TabularLoader
    with pytest.raises(AttributeError):
        Loader.test_split(method='some other str')
    with pytest.raises(AssertionError):
        Loader.test_split(method='explicit')
    with pytest.raises(AssertionError):
        Loader.test_split(method='percentage')


@pytest.mark.dependency(depends=['test_init'])
def test_train_dataloader(create_example_TabularLoader):
    Loader = create_example_TabularLoader
    train_dataloader = Loader.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.dataloader.DataLoader)


@pytest.mark.dependency(depends=['test_init'])
def test_val_dataloader(create_example_TabularLoader):
    Loader = create_example_TabularLoader
    Loader.val_split(method='random', val_size=0.25)
    val_dataloader = Loader.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.dataloader.DataLoader)


@pytest.mark.dependency(depends=['test_init'])
def test_test_dataloader(create_example_TabularLoader):
    Loader = create_example_TabularLoader
    Loader.test_split(method='random', test_size=0.25)
    test_dataloader = Loader.test_dataloader()
    assert isinstance(test_dataloader, torch.utils.data.dataloader.DataLoader)


@pytest.mark.dependency(depends=['test_init'])
def test_save_load(create_example_TabularLoader, tmp_path):
    Loader = create_example_TabularLoader
    Loader.save(tmp_path / 'exampleLoader.pkg')
    Loader2 = TabularLoader.load(tmp_path / 'exampleLoader.pkg')
    assert isinstance(Loader2, TabularLoader)


@pytest.mark.dependency(depends=['test_init'])
def test_read_from_file(tmp_path, create_random_df):
    features = create_random_df.columns[:2]
    labels = create_random_df.columns[2:]
    create_random_df.to_csv(tmp_path / 'example_df.csv')
    batch = 16
    num_workers = 4

    Loader = TabularLoader.read_from_file(tmp_path / 'example_df.csv', features, labels, batch=batch,
                                          num_workers=num_workers)

    assert isinstance(Loader, TabularLoader), 'Returned class is not TabularLoader'
    assert Loader.lparams.batch == batch, 'Kwargs batch not forwarded correctly'
    assert Loader.lparams.num_workers == num_workers, 'Kwargs num_workers not forwarded correctly'
    assert all(Loader.lparams.features == features), 'Features not initialized correctly'
    assert all(Loader.lparams.labels == labels), 'Labels not initialzed correctly'


@pytest.mark.dependency(depends=['test_init'])
def test_option_class(path, tmp_path):
    # test mutually exclusive of load_dataloader and create_dataloader and necessity to include one
    with pytest.raises(AssertionError):
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(KeyError):
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file.pop('load_dataloader')
        _ = yaml_file.pop('create_dataloader')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])

    # test load keys
    with pytest.raises(AssertionError):
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        yaml_file['dataloader']['load_dataloader'] = {}
        _ = yaml_file['dataloader'].pop('create_dataloader')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(TypeError):
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        yaml_file['dataloader']['load_dataloader']['path'] = str(tmp_path / 'TabularLoader.pkg')
        _ = yaml_file['dataloader'].pop('create_dataloader')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])

    # test create keys
    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader'].pop('raw_data_path')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader'].pop('features')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader'].pop('labels')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])

    with pytest.raises(AssertionError):  # load_data and split_data in validation data mutually exclusive
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader']['test_data'].pop('load_data')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])

    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data']['load_data'].pop('path')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data'].pop('split_data')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data']['split_data'].pop('method')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data'].pop('load_data')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])
    with pytest.raises(AssertionError):  # required arg missing
        yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
        _ = yaml_file['dataloader'].pop('load_dataloader')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data']['split_data'].pop('params')
        _ = yaml_file['dataloader']['create_dataloader']['validation_data'].pop('load_data')
        TabularLoader.read_from_yaml(yaml_file['dataloader'])


@pytest.mark.dependency(depends=['test_init', 'test_save_load'])
def test_read_from_yaml_load(path, tmp_path, create_example_TabularLoader):
    # test load functionality
    create_example_TabularLoader.save(tmp_path / 'exampleTabularLoader.pkl')
    yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
    _ = yaml_file['dataloader'].pop('create_dataloader')
    yaml_file['dataloader']['load_dataloader']['path'] = str(tmp_path / 'exampleTabularLoader.pkl')
    dataLoader = TabularLoader.read_from_yaml(yaml_file['dataloader'], batch=16, num_workers=2)
    assert dataLoader.lparams.batch == 16, 'Batch overwrite does not succeed'
    assert dataLoader.lparams.num_workers == 2, 'Num_workers overwrite does not succeed'


@pytest.mark.dependency(depends=['test_init', 'test_read_from_file', 'test_add_val_data',
                                 'test_val_split', 'test_add_test_data', 'test_test_split'])
def test_read_from_yaml_create(path, tmp_path, create_random_df):
    # test create functionality
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
    yaml_file['dataloader']['create_dataloader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    _ = yaml_file['dataloader'].pop('load_dataloader')
    _ = yaml_file['dataloader']['create_dataloader']['validation_data'].pop('load_data')
    _ = yaml_file['dataloader']['create_dataloader']['test_data'].pop('load_data')
    yaml_file['dataloader']['create_dataloader']['save_loader']['path'] = str(tmp_path / 'example_TabularLoader.pkl')
    dataLoader = TabularLoader.read_from_yaml(yaml_file['dataloader'], batch=16, num_workers=2)
    assert dataLoader.lparams.batch == 16, 'Batch overwrite does not succeed'
    assert dataLoader.lparams.num_workers == 2, 'Num_workers overwrite does not succeed'

    # create dataloader without validation data
    yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
    _ = yaml_file['dataloader'].pop('load_dataloader')
    _ = yaml_file['dataloader']['create_dataloader'].pop('validation_data')
    _ = yaml_file['dataloader']['create_dataloader'].pop('save_loader')
    _ = yaml_file['dataloader']['create_dataloader']['test_data'].pop('load_data')
    yaml_file['dataloader']['create_dataloader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    dataLoader = TabularLoader.read_from_yaml(yaml_file['dataloader'])
    assert dataLoader.x_val is None, 'DataLoader intended to not have a validation set, but x_val not None'

    # create dataloader without test data
    yaml_file = parse_yaml(path / 'TabularLoaderEval.yaml')
    _ = yaml_file['dataloader'].pop('load_dataloader')
    _ = yaml_file['dataloader']['create_dataloader'].pop('test_data')
    _ = yaml_file['dataloader']['create_dataloader'].pop('save_loader')
    _ = yaml_file['dataloader']['create_dataloader']['validation_data'].pop('load_data')
    yaml_file['dataloader']['create_dataloader']['raw_data_path'] = str(tmp_path / 'example_samples.csv')
    dataLoader = TabularLoader.read_from_yaml(yaml_file['dataloader'])
    assert dataLoader.x_test is None, 'DataLoader intended to not have a test set, but x_test not None'


@pytest.mark.dependency(depends=['test_init', 'test_read_from_yaml_create'])
def test_read_from_checkpoint(tmp_path, create_TabularLoader, create_LightningFlexMLP, create_random_df):
    # dataLoader
    dataLoader = create_TabularLoader
    dataLoader.lparams.data_path = tmp_path / 'example_samples.csv'
    create_random_df.to_csv(tmp_path / 'example_samples.csv')

    # define model
    model = create_LightningFlexMLP
    model.hparams_update(update_dict={'lparams': dataLoader.lparams})

    # train model for one epoch and save checkpoint
    checkpoint_callback = Checkpointing(filepath=tmp_path / 'test')
    trainer = pl.Trainer(max_epochs=3, checkpoint_callback=checkpoint_callback, logger=False)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())

    # check if TabularLoader can be restored
    TabularLoader.read_from_checkpoint(tmp_path / 'test.ckpt')



