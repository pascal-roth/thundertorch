#######################################################################################################################
# test functionalities of utils.general.py
#######################################################################################################################

# import packages
import argparse
import pytest
import pandas as pd
import pytorch_lightning as pl
from pathlib import PosixPath

from .MinimalLightningModel import MinimalLightningModule

from thunder_torch.loader import TabularLoader
from thunder_torch.models import LightningFlexMLP
from thunder_torch.callbacks import Checkpointing
from thunder_torch.utils.general import load_model_from_checkpoint, dynamic_imp, run_model


def test_load_model_from_checkpoint(tmp_path: PosixPath, create_TabularLoader: TabularLoader,
                                    create_LightningFlexMLP: LightningFlexMLP,
                                    create_random_df: pd.DataFrame) -> None:
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

        model2_restored = load_model_from_checkpoint(tmp_path / 'test_general_load_from_ckpt_fail.ckpt')