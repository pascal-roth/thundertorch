import pytest
import argparse
import os

import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP


@pytest.mark.dependency()
def test_init(create_TabularLoader):
    Loader = create_TabularLoader
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [16, 16]})
    model = LightningFlexMLP(hparams)
    trainer = pl.Trainer(fast_dev_run=True, logger=False)
    trainer.fit(model, train_dataloader=Loader.train_dataloader(), val_dataloaders=Loader.val_dataloader())
    trainer.test(model, test_dataloaders=Loader.test_dataloader())


@pytest.mark.dependency(depends=['test_init'])
def test_check_hparams():
    # check of hparams without default
    with pytest.raises(AssertionError):  # 'n_inp' missing
        hparams = argparse.Namespace(**{'n_out': 3, 'hidden_layer': [16, 16]})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # 'n_out' missing
        hparams = argparse.Namespace(**{'n_inp': 3, 'hidden_layer': [16, 16]})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # 'hidden_layer' missing
        hparams = argparse.Namespace(**{'n_out': 3, 'n_inp': 3})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # 'n_int' wrong type
        hparams = argparse.Namespace(**{'n_inp': 3.0, 'n_out': 3, 'hidden_layer': [16, 16]})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # 'n_out' wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3.0, 'hidden_layer': [16, 16]})
        LightningFlexMLP(hparams)
    with pytest.raises(TypeError):  # 'hidden_layer' wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, '5']})
        LightningFlexMLP(hparams)

    # check of activation, loss, optimizer and scheduler function
    with pytest.raises(AssertionError):  # activation function wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'activation': None})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # activation function not included in torch
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'activation': 'adam'})
        LightningFlexMLP(hparams)

    with pytest.raises(AssertionError):  # loss function wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'loss': 1.0})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # loss function not included in torch
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'loss': 'mse'})
        LightningFlexMLP(hparams)

    with pytest.raises(AssertionError):  # optimizer params dict empty
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'optimizer': {}})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # optimizer function wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'optimizer': {'type': None}})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # optimizer function not included in torch
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'optimizer': {'type': 'ad'}})
        LightningFlexMLP(hparams)

    with pytest.raises(AssertionError):  # scheduler params dict empty
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'scheduler': {}})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # scheduler function wrong type
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'scheduler':
            {'execute': True, 'type': None}})
        LightningFlexMLP(hparams)
    with pytest.raises(AssertionError):  # scheduler function not included in torch
        hparams = argparse.Namespace(**{'n_inp': 3, 'n_out': 3, 'hidden_layer': [16, 16], 'scheduler':
            {'execute': True, 'type': 'some function'}})
        LightningFlexMLP(hparams)


@pytest.mark.dependency(depends=['test_init'])
def test_default_adjustment():
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [16, 16], 'batch': 16, 'num_workers': 4,
                                    'output_relu': True})
    model = LightningFlexMLP(hparams)
    assert model.hparams.batch == 16, 'Passed batch value has been overwritten by default!'
    assert model.hparams.num_workers == 4, 'Passed num_workers value has been overwritten by default!'
    assert model.hparams.output_relu is True, 'Passed output_relu boolean has been overwritten by default!'


@pytest.mark.dependency(depends=['test_init'])
def test_hparams_save(tmp_path):
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [16, 16]})
    model = LightningFlexMLP(hparams)
    model.hparams_save(tmp_path / 'hparams.yaml')
    assert os.path.isfile(tmp_path / 'hparams.yaml')


@pytest.mark.dependency(depends=['test_init'])
def test_hparams_update():
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [16, 16]})
    model = LightningFlexMLP(hparams)
    update = {'n_out': 3, 'output_relu': True}
    model.hparams_update(update)
    assert model.hparams.n_out == 3
    assert model.hparams.output_relu is True

