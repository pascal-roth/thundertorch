#######################################################################################################################
# Use Constant-Volume and Fixed-Mass Homogeneous-Reactor Model for integration test of FlexMLP_pl
#######################################################################################################################

# import packages
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from thunder_torch.utils import train_config
from thunder_torch.models import LightningFlexMLP
from thunder_torch.loader import TabularLoader


# @pytest.mark.dependency(depends=["./test_LightningFlexMLP_unit.py::test_init"], scope='session')
# @pytest.mark.dependency(depends=["../utils/test_training.py::test_config_deterministic"])
def test_LightningFlexMLP_integration() -> None:
    argsTrainer = train_config({'deterministic': True}, {'params': {}})
    np.random.seed(0)

    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 1, 'hidden_layer': [4, 4]})
    model = LightningFlexMLP(hparams)

    train = pd.DataFrame(np.random.rand(100, 3), columns=['x1', 'x2', 'y'])
    x_val = pd.DataFrame(np.random.rand(20, 2), columns=['x1', 'x2'])
    y_val = pd.DataFrame(np.random.rand(20, 1), columns=['y'])

    argsLoader = {'df_samples': train, 'features': ['x1', 'x2'], 'labels': ['y']}
    dataLoader = TabularLoader(**argsLoader, batch=model.hparams.batch, num_workers=model.hparams.num_workers)

    dataLoader.x_val = x_val
    dataLoader.y_val = y_val

    trainer = pl.Trainer(**argsTrainer, max_epochs=10, logger=False)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(shuffle=False),
                val_dataloaders=dataLoader.val_dataloader(shuffle=False))

    # compare training, validation and test loss
    val_losses = trainer.tng_tqdm_dic
    assert np.round(val_losses['val_loss'], decimals=5) == 0.08063, {'Integration test failed'}
    assert val_losses['loss'] == '0.105', {'Integration test failed'}
