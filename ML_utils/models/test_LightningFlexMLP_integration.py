#######################################################################################################################
# Use Constant-Volume and Fixed-Mass Homogeneous-Reactor Model for integration test of FlexMLP_pl
#######################################################################################################################

# import packages
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import cantera as ct
import pytorch_lightning as pl
import multiprocessing as mp
import itertools
import pytest

from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP
from stfs_pytoolbox.ML_Utils.loader import TabularLoader

ct.suppress_thermo_warnings()
gas_multi = {}


def init_process():
    gas_multi[0] = ct.Solution('gri30.xml')
    gas_multi[0].transport_model = 'Multi'


def homogeneous_reactor(argsReactor):
    """
    Constant-volume and fixed-mass homogeneous reactor model

    Parameters
    ----------
    equivalence_ratio       - float
    reactorPressure         - float
    reactorTemperature      - float

    Returns
    -------
    values                  - np.array
    """
    equivalence_ratio, reactorPressure, reactorTemperature = argsReactor
    gas = gas_multi[0]

    # Create Reactor
    gas.TP = reactorTemperature, reactorPressure
    gas.set_equivalence_ratio(equivalence_ratio, 'CH4:1.0', 'O2:1.0, N2:3.76')
    r1 = ct.IdealGasReactor(contents=gas, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])

    #  Solution of reaction
    time = 0.0
    t_step = 5.e-8
    t_end = 0.125 * 1.e-3

    values = np.zeros((int(t_end / t_step), 5))

    while time < t_end:
        if time == 0.0:
            n = 0
        else:
            n += 1

        # Calculate the reactor parameters for the point in time
        time += t_step
        sim.advance(time)

        # Summarize all values to be saved in an array
        values[n] = (reactorTemperature, time, r1.thermo.T, r1.Y[gas.species_index('CO2')],
                     r1.thermo.net_production_rates[gas.species_index('H2O')])

    return values


def generate_samples():
    """
    iterator over different temperatures

    Parameters
    ----------
    temperature     - list of temperature values

    Returns
    -------
    samples         - pd.DataFrame including example samples
    """
    # set parameters
    equivalence_ratio = 1.0
    pressure = 20 * ct.one_atm
    temperature = [1500, 1550, 1575, 1600]
    samples = np.zeros((10000, 5))
    n = 0

    pool = mp.Pool(processes=4, initializer=init_process, initargs=())
    values = pool.map(homogeneous_reactor, zip(itertools.repeat(equivalence_ratio), itertools.repeat(pressure),
                                               temperature))

    for i in range(len(values)):
        # separate the list of all temperatures into the single ones
        values_run = values[i]
        samples[n:(n+2500), :] = values_run
        n += 2500

    pool.close()

    samples = pd.DataFrame(samples)
    samples.columns = ['T_0', 'time', 'T', 'yCO2', 'wH2O']

    return samples


# @pytest.mark.dependency(depends=["./test_LightningFlexMLP_unit.py::test_init"], scope='session')
def test_LightningFlexMLP_integration():
    hparams = argparse.Namespace(**{'n_inp': 2, 'n_out': 3, 'hidden_layer': [64, 64]})
    model = LightningFlexMLP(hparams)

    # initialize weights constant (necessary for integration test)
    for layer in model.layers:
        torch.nn.init.constant_(layer.weight, val=0.1)
        torch.nn.init.constant_(layer.bias, val=0.1)
    torch.nn.init.constant_(model.output.weight, val=0.1)
    torch.nn.init.constant_(model.output.bias, val=0.1)

    data = generate_samples()
    argsLoader = {'df_samples': data, 'features': ['T_0', 'time'], 'labels': ['T', 'yCO2', 'wH2O'],
                  'val_split': {'method': 'explicit', 'params': {'T_0': 1550}},
                  'test_split': {'method': 'explicit', 'params': {'T_0': 1575}}}
    dataLoader = TabularLoader(**argsLoader, batch=model.hparams.batch, num_workers=model.hparams.num_workers)

    trainer = pl.Trainer(max_epochs=10, logger=False)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(shuffle=False),
                val_dataloaders=dataLoader.val_dataloader(shuffle=False))
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader(shuffle=False))

    # compare training, validation and test loss
    val_losses = trainer.tng_tqdm_dic
    assert np.round(val_losses['val_loss'], decimals=5) == 0.11361, {'Integration test failed'}
    assert val_losses['loss']== '0.050', {'Integration test failed'}
