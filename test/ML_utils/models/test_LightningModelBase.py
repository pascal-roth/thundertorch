import pytest
import os
import argparse
from pathlib import PosixPath

from thunder_torch.models import LightningModelBase, LightningFlexMLP


def test_hparams_save(tmp_path: PosixPath) -> None:
    model = LightningModelBase()
    model.hparams_save(tmp_path / 'hparams.yaml')
    assert os.path.isfile(tmp_path / 'hparams.yaml')


def test_hparams_update() -> None:
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 2, 'hidden_layer': [16, 16]})
    model = LightningFlexMLP(hparams)
    update = {'n_out': 3, 'output_activation': 'LogSigmoid'}
    model.hparams_update(update)
    assert model.hparams.n_out == 3
    assert model.hparams.output_activation == 'LogSigmoid'
