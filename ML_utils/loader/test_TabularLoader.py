import pytest
import torch
import yaml
import argparse
import os
from pathlib import Path
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.loader import TabularLoader


@pytest.fixture(scope='module')
def create_example_TabularLoader(create_random_df):
    return TabularLoader(create_random_df, features=['T_0', 'P_0'], labels=['yCO2', 'wH2O'])


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


@pytest.mark.dependency(depends=['test_init', 'test_read_from_file', 'test_save_load', 'test_add_val_data',
                                 'test_val_split', 'test_add_test_data', 'test_test_split'])
def test_read_from_yaml(tmp_path, create_random_df, create_example_TabularLoader):
    path = Path(__file__).resolve()
    path = path.parents[0]
    # test source flag
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader'].pop('source')
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['DataLoader']['source'] = 'some other str'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))

    # test load functionality
    create_example_TabularLoader.save(tmp_path / 'exampleTabularLoader.pkl')
    yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
    yaml_file['DataLoader']['source'] = 'load'
    yaml_file['DataLoader']['load_DataLoader']['path'] = tmp_path / 'exampleTabularLoader.pkl'
    dataLoader = TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']), batch=16, num_workers=2)
    assert dataLoader.lparams.batch == 16, 'Batch overwrite does not succeed'
    assert dataLoader.lparams.num_workers == 2, 'Num_workers overwrite does not succeed'

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader'].pop('load_DataLoader')
        yaml_file['DataLoader']['source'] = 'load'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['DataLoader']['load_DataLoader'] = {}
        yaml_file['DataLoader']['source'] = 'load'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(TypeError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        yaml_file['DataLoader']['source'] = 'load'
        yaml_file['DataLoader']['load_DataLoader']['path'] = tmp_path / 'TabularLoader.pkg'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))

    # test create functionality
    create_random_df.to_csv(tmp_path / 'example_samples.csv')
    yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
    yaml_file['DataLoader']['source'] = 'create'
    yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
    dataLoader = TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']), batch=16, num_workers=2)
    assert dataLoader.lparams.batch == 16, 'Batch overwrite does not succeed'
    assert dataLoader.lparams.num_workers == 2, 'Num_workers overwrite does not succeed'

    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader'].pop('create_DataLoader')
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader'].pop('raw_data_path')
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader'].pop('features')
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader'].pop('labels')
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader'].pop('validation_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['validation_data'].pop('source')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(TypeError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['validation_data']['source'] = 'some other str'
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['validation_data']['source'] = 'load'
        _ = yaml_file['DataLoader']['create_DataLoader']['validation_data'].pop('load_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['validation_data'].pop('split_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader'].pop('test_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['test_data'].pop('source')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(TypeError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['test_data']['source'] = 'some other str'
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['test_data']['source'] = 'load'
        _ = yaml_file['DataLoader']['create_DataLoader']['test_data'].pop('load_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))
    with pytest.raises(AssertionError):
        yaml_file = yaml.load(open(path / 'TabularLoaderEval.yaml'), Loader=yaml.FullLoader)
        _ = yaml_file['DataLoader']['create_DataLoader']['test_data'].pop('split_data')
        yaml_file['DataLoader']['create_DataLoader']['raw_data_path'] = tmp_path / 'example_samples.csv'
        TabularLoader.read_from_yaml(argparse.Namespace(**yaml_file['DataLoader']))


@pytest.mark.dependency(depends=['test_init', 'test_read_from_yaml'])
def test_read_from_checkpoint(tmp_path, create_TabularLoader, create_LightningFlexMLP, create_random_df):
    # dataLoader
    dataLoader = create_TabularLoader
    dataLoader.lparams.data_path = tmp_path / 'example_samples.csv'
    create_random_df.to_csv(tmp_path / 'example_samples.csv')

    # define model
    model = create_LightningFlexMLP
    model.hparams_update(update_dict=dataLoader.lparams)

    # train model for one epoch and save checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tmp_path / 'test')
    trainer = pl.Trainer(max_epochs=3, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())

    # find checkpoint file
    path = None
    for file in os.listdir(tmp_path):
        if file.endswith(".ckpt"):
             path = os.path.join(tmp_path, file)

    # check if TabularLoader can be restored
    TabularLoader.read_from_checkpoint(path)

