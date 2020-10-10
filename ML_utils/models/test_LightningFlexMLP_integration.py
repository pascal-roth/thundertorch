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
import pytest
import multiprocessing as mp
import itertools

from stfs_pytoolbox.ML_Utils.models import LightningFlexMLP
from stfs_pytoolbox.ML_Utils.loader import TabularLoader

ct.suppress_thermo_warnings()
pode_multi = {}


def init_process():
    pode_multi[0] = ct.Solution('cai_ome14_2019.xml')
    pode_multi[0].transport_model = 'Multi'


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
    pode = pode_multi[0]

    # Create Reactor
    pode.TP = reactorTemperature, reactorPressure
    pode.set_equivalence_ratio(equivalence_ratio, 'OME3', 'O2:0.21 N2:0.79')
    r1 = ct.IdealGasReactor(contents=pode, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])

    #  Solution of reaction
    time = 0.0
    t_step = 1.e-7
    t_end = 0.25 * 1.e-3

    values = np.zeros((int(t_end / t_step), 5))

    # Parameters of PV
    OME3_0 = r1.Y[pode.species_index('OME3')]

    while time < t_end:
        if time == 0.0:
            n = 0
        else:
            n += 1

        # Calculate the reactor parameters for the point in time
        time += t_step
        sim.advance(time)

        # Calculate the PV
        PV = r1.Y[pode.species_index('H2O')] * 0.5 + r1.Y[pode.species_index('CH2O')] * 0.5 + \
             (- r1.Y[pode.species_index('OME3')] + OME3_0) * 0.5 + r1.Y[pode.species_index('CO2')] * 0.05

        # Summarize all values to be saved in an array
        values[n] = (reactorTemperature, PV, r1.thermo.T, r1.Y[pode.species_index('CO2')],
                     r1.thermo.net_production_rates[pode.species_index('H2O')])

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
    temperature = [930, 940, 945, 950]
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
    samples.columns = ['T_0', 'PV', 'T', 'yCO2', 'wH2O']

    return samples


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
    argsLoader = {'df_samples': data, 'features': ['T_0', 'PV'], 'labels': ['T', 'yCO2', 'wH2O'],
                  'val_split': {'method': 'explicit', 'val_params': {'T_0': 940}},
                  'test_split': {'method': 'explicit', 'test_params': {'T_0': 945}}}
    dataLoader = TabularLoader(**argsLoader, batch=model.hparams.batch, num_workers=model.hparams.num_workers)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(shuffle=False), val_dataloaders=dataLoader.val_dataloader(shuffle=False))
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader(shuffle=False))

    # compare training, validation and test loss
    val_losses = trainer.progress_bar_metrics
    assert np.round(val_losses['val_loss'], decimals=5) == 0.08022, {'Integration test failed'}
