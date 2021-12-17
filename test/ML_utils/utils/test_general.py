#######################################################################################################################
# test functionalities of utils.general.py
#######################################################################################################################

# import packages
import argparse
import os
import torch
import pytest
import pytorch_lightning as pl
from pathlib import PosixPath, Path

from thunder_torch.loader import TabularLoader
from thunder_torch.models import LightningFlexMLP
from thunder_torch.callbacks import Checkpointing
from thunder_torch.utils.general import load_model_from_checkpoint, get_ckpt_path, dynamic_imp, run_model


@pytest.fixture(scope='module')
def path() -> Path:
    path = Path(__file__).resolve()
    return path.parents[0]

@pytest.mark.dependency()
def test_get_ckpt_path(tmp_path: PosixPath, create_TabularLoader: TabularLoader,
                       create_LightningFlexMLP: LightningFlexMLP) -> None:
    # define model
    model = create_LightningFlexMLP

    # train model for one epoch and save checkpoint
    os.makedirs(tmp_path / 'test_general_get_ckpt_path')
    checkpoint_callback = Checkpointing(filepath=tmp_path / 'test_general_get_ckpt_path/ckpt_path')
    trainer = pl.Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback, logger=False)
    trainer.fit(model, train_dataloader=create_TabularLoader.train_dataloader(),
                val_dataloaders=create_TabularLoader.val_dataloader())

    # check if exact path is given path when entered to network
    path1 = get_ckpt_path(tmp_path / 'test_general_get_ckpt_path/ckpt_path.ckpt')
    assert path1 == str(tmp_path / 'test_general_get_ckpt_path/ckpt_path.ckpt')

    # check if path to network is constructed if directory is given
    path2 = get_ckpt_path(tmp_path / 'test_general_get_ckpt_path')
    assert path2 == str(tmp_path / 'test_general_get_ckpt_path/ckpt_path.ckpt')

    # entered path does not exists
    with pytest.raises(AttributeError):
        get_ckpt_path(tmp_path / 'some_random_path.ckpt')

    # two checkpoint files in given directory
    with pytest.raises(AssertionError):
        checkpoint_callback = Checkpointing(filepath=tmp_path / 'test_general_get_ckpt_path/ckpt_path2')
        trainer = pl.Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback, logger=False)
        trainer.fit(model, train_dataloader=create_TabularLoader.train_dataloader(),
                    val_dataloaders=create_TabularLoader.val_dataloader())

        get_ckpt_path(tmp_path / 'test_general_get_ckpt_path')


@pytest.mark.dependency(depends=['test_get_ckpt_path'])
def test_load_model_from_checkpoint(tmp_path: PosixPath, create_TabularLoader: TabularLoader,
                                    create_LightningFlexMLP: LightningFlexMLP) -> None:
    # define model
    model = create_LightningFlexMLP

    # train model for one epoch and save checkpoint
    checkpoint_callback = Checkpointing(filepath=tmp_path / 'test_general_load_from_ckpt')
    trainer = pl.Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback, logger=False)
    trainer.fit(model, train_dataloader=create_TabularLoader.train_dataloader(),
                val_dataloaders=create_TabularLoader.val_dataloader())

    model_restored = load_model_from_checkpoint(tmp_path / 'test_general_load_from_ckpt.ckpt')
    assert model_restored.hparams.model_type == 'LightningFlexMLP', 'Loading failed'
    assert model_restored.hparams.n_inp == 2, 'Loading failed'

    # input type of function not a string
    with pytest.raises(TypeError):
        load_model_from_checkpoint(3)  # type: ignore

    # model type cannot be found in _modules_models
    with pytest.raises(NameError):
        # define model without model_type
        model.hparams.model_type = 'LightningFlexMlp'

        # train model for one epoch and save checkpoint
        checkpoint_callback = Checkpointing(filepath=tmp_path / 'test_general_load_from_ckpt_fail')
        trainer = pl.Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback, logger=False)
        trainer.fit(model, train_dataloader=create_TabularLoader.train_dataloader(),
                    val_dataloaders=create_TabularLoader.val_dataloader())

        load_model_from_checkpoint(tmp_path / 'test_general_load_from_ckpt_fail.ckpt')


def test_dynamic_imp(tmp_path: PosixPath, create_TabularLoader: TabularLoader,
                     create_LightningFlexMLP: LightningFlexMLP, path: Path) -> None:

    _, model_cls = dynamic_imp(str(path / "imported"), "LightningFlexMLPImported")
    model = model_cls(argparse.Namespace(**{"inputs": 2, "outputs": 2, "number_hidden_layers": [300, 300]}))

    assert isinstance(model, pl.LightningModule), "model import failed"
    assert model.hparams.inputs == 2,  "model import failed"

    # trigger error when wrong path to python file given
    with pytest.raises(ImportError):
        dynamic_imp("some_random_name.py", "LightningFlexMLPImported")

    # trigger error when wrong path to python file given
    with pytest.raises(AttributeError):
        dynamic_imp(str(path / "imported"), "some_random_name")
