#######################################################################################################################
# Use Constant-Volume and Fixed-Mass Homogeneous-Reactor Model for integration test of FlexMLP_pl
#######################################################################################################################

# import packages
import argparse
import numpy as np
import pandas as pd
import cantera as ct
import pytorch_lightning as pl
from pathlib import Path
from TabularLoader import TabularLoader
from flexMLP_pl import flexMLP_pl
ct.suppress_thermo_warnings()

def homogeneous_reactor(equivalence_ratio, reactorPressure, reactorTemperature):
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
    pode = ct.Solution('cai_ome14_2019.xml')

    # Create Reactor
    pode.TP = reactorTemperature, reactorPressure
    pode.set_equivalence_ratio(equivalence_ratio, 'OME3', 'O2:0.21 N2:0.79')
    r1 = ct.IdealGasReactor(contents=pode, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])

    #  Solution of reaction
    time = 0.0
    t_step = 1.e-7
    t_end = 0.25 * 1.e-3

    values = np.zeros((int(t_end / t_step), 9))

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

        HRR = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies / r1.mass)

        # Summarize all values to be saved in an array
        values[n] = (equivalence_ratio, reactorTemperature, PV, HRR, r1.thermo.T, r1.thermo.P,
                     r1.Y[pode.species_index('OME3')], r1.Y[pode.species_index('CO2')],
                     r1.thermo.net_production_rates[pode.species_index('H2O')])

    return values


def generate_samples(temperature):
    """
    iterator over different temperatures

    Parameters
    ----------
    temperature     - list of temperature values

    Returns
    -------
    samples         - pd.DataFrame including example samples
    """
    if temperature == [930, 940, 945, 950]:
        path = Path(__file__).resolve()
        path_sample = path.parents[0] / 'example_samples.csv'.format(temperature)
        samples = pd.read_csv(path_sample)
    else:
        # set parameters
        equivalence_ratio = 1.0
        pressure = 20 * ct.one_atm

        for i, temperature_run in enumerate(temperature):
            samples_run = homogeneous_reactor(equivalence_ratio, pressure, temperature_run)
            if i == 0:
                samples = samples_run
            else:
                samples = np.append(samples, samples_run, axis=0)

        samples = pd.DataFrame(samples)
        samples.columns = ['phi', 'T_0', 'PV', 'HRR', 'T', 'P', 'yOME3', 'yCO2', 'wH2O']

        save_samples = False
        if save_samples:
            path = Path(__file__).resolve()
            path_sample = path.parents[0] / 'example_samples.csv'.format(temperature)
            samples.to_csv(path_sample)

    return samples


def parseArguments():
    # hyperparameters to build model ##################################################################################
    hparams_parser = argparse.ArgumentParser()

    # Add mutually_exclusive_group to either load a FlexMLP model or create on based on input
    hparams_parser.add_argument('-f', '--features', nargs='+', default=['T_0', 'PV'],
                                help="Provide list of features extracted from training data")
    hparams_parser.add_argument('-l', '--labels', nargs='+', default=['T', 'yOME3', 'wH2O'],
                                help="Provide list of labels extracted from training data")
    hparams_parser.add_argument('-hl', '--hidden-layer', type=int, nargs='+', default=[64, 64],
                                help="Provide list of hidden layer neurons")

    # hyperparameter of the model
    hparams_parser.add_argument('-b', '--batch', type=int, dest='batch', default=16,
                                help='Batch size during training')
    hparams_parser.add_argument('-lr', '--learning-rate', type=float, dest='lr', default=1e-3,
                                help='Learning rate for optimizer')
    hparams_parser.add_argument('-r', '--output-relu', action='store_true',
                                help="Adds a relu activation function to output layer")
    hparams_parser.add_argument('-a', '--activation', choices=['relu', 'tanh', 'softplus'], default='relu',
                                help="Defines activation function for each layer")
    hparams_parser.add_argument('-lo', '--loss', choices=['MSE', 'RelativeMSE'], default='MSE',
                                help="Defines loss function for training")
    hparams_parser.add_argument('-o', '--optimizer', choices=['adam'], default='adam',
                                help="Defines optimizer function for network training")
    hparams_parser.add_argument('-s', '--scheduler', action='store_true', default=False,
                                help="Flag if learning rate scheduler should be used")
    hparams_parser.add_argument('-nw', '--num_workers', type=int, default=4,
                                help="Define number of workers in DataLoaders to increase training speed")

    hparams = hparams_parser.parse_args()

    # training parameters #############################################################################################
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument('-g', '--gpu', type=int, default=0,
                              help="Use GPU of given ID, default=0")
    train_parser.add_argument('-w', '--init-weights', dest='weight', type=float, default=0.1,
                              help='initializes weights of last layer with given number')
    train_parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=10,
                              help='Number of epochs to train')
    train_parser.add_argument('--output', '-out', dest='output', required=False, default='TorchModel.pt',
                              help='file name of best model that is saved')

    args = train_parser.parse_args()

    return hparams, args


if __name__ == '__main__':

    hparams, args = parseArguments()
    hparams.x_scaler = None
    hparams.y_scaler = None

    # generate training, validation and test samples as one DF
    print('Start to generate sample set ...')
    data = generate_samples([930, 940, 945, 950])
    print('DONE!')

    print('Split data:')
    Loader = TabularLoader(data)
    Loader.val_split(method='sample', split_method='explicit', val_params={'T_0': 940})
    Loader.test_split(method='sample', split_method='explicit', test_params={'T_0': 945})
    print('DONE!')

    print('Construct Network and start training:')
    model = flexMLP_pl(hparams=hparams, TabularLoader=Loader)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model)

    # compare training, validation and test loss
    # TODO: after complete logging add loss comparison
