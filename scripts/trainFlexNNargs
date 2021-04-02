#######################################################################################################################
# Create flexMLP by parsing args
#######################################################################################################################

# import packages
import argparse
import os
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger

from thunder_torch.models.LightningFlexMLP import LightningFlexMLP
from thunder_torch.loader.TabularLoader import TabularLoader


def parseArguments():
    """
    Parse Arguments to load data, load/construct LightningModule and train model with data

    Returns
    -------
    hparams     - Namespace object including model hyperparameters
    args        - Namespace object including location paths and training arguments
    """
    # hyperparameters to build model ##################################################################################
    hparams_parser = argparse.ArgumentParser()  # TODO: combine two argsparse functions

    # Add mutually_exclusive_group to either load a FlexMLP model or create on based on input
    group = hparams_parser.add_argument_group("Mutally exclusive group to load or create FlexMLP model")
    group.add_argument('--model', '-m', dest='model',
                       help='PyTorch model to load from .pt file')
    group.add_argument('-f', '--features', nargs='+',
                       help="Provide list of features extracted from training data")
    group.add_argument('-l', '--labels', nargs='+',
                       help="Provide list of labels extracted from training data")
    group.add_argument('-hl', '--hidden-layer', type=int, nargs='+',
                       help="Provide list of hidden layer neurons")

    # hyperparameter of the model
    hparams_parser.add_argument('-b', '--batch', type=int, dest='batch', default=16,
                                help='Batch size during training')
    hparams_parser.add_argument('-lr', '--learning-rate', type=float, dest='lr', default=1e-3,
                                help='Learning rate for optimizer')
    hparams_parser.add_argument('-r', '--output_relu', action='store_true',
                                help="Adds a relu activation function to output layer")
    hparams_parser.add_argument('-a', '--activation', choices=['relu', 'tanh', 'softplus'], default='relu',
                                help="Defines activation function for each layer")
    hparams_parser.add_argument('-lo', '--loss', choices=['MSE', 'RelativeMSE'], default='MSE',
                                help="Defines loss function for training")
    hparams_parser.add_argument('-o', '--optimizer', choices=['adam'], default='adam',
                                help="Defines optimizer function for network training")
    hparams_parser.add_argument('-s', '--scheduler', action='store_true', default=False,
                                help="Flag if learning rate scheduler should be used")
    hparams_parser.add_argument('-nw', '--num_workers', type=int, default=10,
                                help="Define number of workers in DataLoaders to increase training speed")

    # figure out which model to use
    # parser.add_argument('--model_name', type=str, default='MLP', help='MLP or CNN')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    # temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    # if temp_args.model_name == 'MLP':
    #     parser = LightningFlexMLP.add_model_specific_args(parser)
    # elif temp_args.model_name == 'CNN':
    #    parser = LightningFlexNN.add_model_specific_args(parser)

    hparams = hparams_parser.parse_args()

    # Make model and model definition mutually_exclusive, there is not way to do it in with argparse
    if hasattr(hparams, 'model') and not (hasattr(hparams, 'features') or hasattr(hparams, 'labels') or hasattr(hparams, 'hidden_layer')):
        hparams_parser.error("The following arguments are mutally exclusive:"
                             " [-m/--model] or [-f/--features, -l/--labels, -hl/--hidden-layer]")
    elif all(hasattr(hparams, elem) for elem in ['features', 'labels', 'hidden_layer']) and not hasattr(hparams, 'model'):
        hparams_parser.error("The following arguments are required together: "
                             "[-f/--features, -l/--labels, -hl/--hidden-layer]")

    # training parameters #############################################################################################
    train_parser = argparse.ArgumentParser()
    train_parser = pl.Trainer.add_argparse_args(train_parser)
    train_parser.add_argument('-d', '--data', type=str, dest='data', required=True,
                              help='Location of file to load for network data. Allows are Loaders saved as .pkg, '
                                   'checkpoint with necessary information and files of type .txt, .csv, .ulf, .h5'
                                   '(Delimiter must be spaces)')
    # train_parser.add_argument('-g', '--gpu', type=int, default=0,
    #                           help="Use GPU of given ID, default=0")
    train_parser.add_argument('-w', '--init-weights', dest='weight', type=float, default=0.1,
                              help='initializes weights of last layer with given number')
    # train_parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=10,
    #                           help='Number of epochs to train')
    train_parser.add_argument('--output', '-out', dest='output', required=False, default='TorchModel.pt',
                              help='file name of best model that is saved')

    args = train_parser.parse_args()

    return hparams, args


def main(hparams, args):
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    hparams         - Namespace object including arguments to load/construct model
    args            - Namespace object including location paths and training arguments
    """
    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')  # TODO: Discuss with Julian if necessary
    lr_logger = LearningRateLogger( )

    # # pick model
    # if args.model_name == 'MLP':
    #     model = LightningFlexMLP(hparams=args)
    # elif args.model_name == 'CNN':
    #     model = LightningFlexNN(hparams=args)

    hparams.n_inp = len(hparams.features)
    hparams.n_out = len(hparams.labels)

    # create or restore model
    if hasattr(hparams, 'model'):
        model = LightningFlexMLP.load_from_checkpoint(hparams.model)
        trainer = pl.Trainer.from_argparse_args(args, resume_from_checkpoint=hparams.model, callbacks=[lr_logger],
                                                early_stop_callback=early_stop_callback)
        logging.info('Model has been loaded from file: {}'.format(hparams.model))

    elif all(hasattr(hparams, elem) for elem in ['features', 'labels', 'hidden_layer']):
        # check for features and labels
        if not isinstance(hparams.features, list): hparams.features = [hparams.features]
        if not isinstance(hparams.labels, list): hparams.features = [hparams.labels]

        model = LightningFlexMLP(hparams)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_logger], early_stop_callback=early_stop_callback)
        logging.info('New model created!')

    logging.info('Start to create/load Loader!')
    _, file_exention = os.path.splitext(args.data)

    if file_exention == '.pkg':
        Loader = TabularLoader.load(args.data, features=model.hparams.features, labels=model.hparams.labels,
                                    batch=model.hparams.batch, num_workers=model.hparams.num_workers)
        Loader.lparams.data_location = args.data
    elif file_exention == '.ckpg':
        Loader = TabularLoader.read_from_checkpoint(args.data, features=model.hparams.features, labels=model.hparams.labels,
                                                    batch=model.hparams.batch, num_workers=model.hparams.num_workers)
    elif file_exention in ['.txt', '.csv', '.ulf', '.h5']:
        Loader = TabularLoader.read_from_file(args.data, features=model.hparams.features, labels=model.hparams.labels,
                                              batch=model.hparams.batch, num_workers=model.hparams.num_workers)

        Loader.val_split(method='random', val_size=0.2)
        Loader.test_split(methdo='random', test_size=0.05)
    else:
        logging.error('Data file type not supported!')
        raise TypeError('Data file type not supported!')

    logging.info('DONE! Start to train and test model!')
    model.hparams_update(update_dict=Loader.lparams)
    trainer.fit(model, train_dataloader=Loader.train_dataloader(), val_dataloaders=Loader.val_dataloader())
    trainer.test(model, test_dataloaders=Loader.test_dataloader())

if __name__ == '__main__':
    hparams, args = parseArguments()

    main(hparams, args)
