#######################################################################################################################
# Create flexMLP by parsing args
#######################################################################################################################

# import packages
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger

from .flexMLP_pl import flexMLP_pl
from .flexCNN_pl import flexCNN_pl
from .TabularLoader import TabularLoader


def parseArguments():
    """
    Parse Arguments to load data, load/construct LightningModule and train model with data

    Returns
    -------
    hparams     - Namespace object including model hyperparameters
    args        - Namespace object including location paths and training arguments
    """
    # hyperparameters to build model ##################################################################################
    hparams_parser = argparse.ArgumentParser()

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
    #     parser = flexMLP_pl.add_model_specific_args(parser)
    # elif temp_args.model_name == 'CNN':
    #    parser = flexCNN_pl.add_model_specific_args(parser)

    hparams = hparams_parser.parse_args()

    # Make model and model definition mutually_exclusive, there is not way to do it in with argparse
    if hparams.model:
        if hparams.features or hparams.labels or hparams.hidden_layer:
            hparams_parser.error("The following arguments are mutally exclusive:"
                                 " [-m/--model] or [-f/--features, -l/--labels, -hl/--hidden-layer]")
    else:
        if not (hparams.features and hparams.labels and hparams.hidden_layer):
            hparams_parser.error("The following arguments are required together: "
                                 "[-f/--features, -l/--labels, -hl/--hidden-layer]")

        # check for features and labels
        if not isinstance(hparams.features, list):
            hparams.features = [hparams.features]
        assert all(isinstance(elem, str) for elem in hparams.features), "Given features is not a list of strings!"

        if not isinstance(hparams.labels, list):
            hparams.features = [hparams.labels]
        assert all(isinstance(elem, str) for elem in hparams.labels), "Given labels is not a list of strings!"

    # training parameters #############################################################################################
    train_parser = argparse.ArgumentParser()
    train_parser = pl.Trainer.add_argparse_args(train_parser)
    train_parser.add_argument('-d', '--data', type=str, dest='data', required=True,
                              help='File to load for network data. Valid extensions are .txt, .csv, .ulf, .h5'
                                   ' Delimiter must be spaces"')
    # train_parser.add_argument('-g', '--gpu', type=int, default=0,
    #                           help="Use GPU of given ID, default=0")
    train_parser.add_argument('-w', '--init-weights', dest='weight', type=float, default=0.1,
                              help='initializes weights of last layer with given number')
    # train_parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=10,
    #                           help='Number of epochs to train')
    train_parser.add_argument('--output', '-out', dest='output', required=False, default='TorchModel.pt',
                              help='file name of best model that is saved')  # TODO: überlegen ob wirklich location defined werden soll oder einfach immer in den lightning_logs

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
    _, file_exention = os.path.splitext(args.data)

    print("Loading training data:")
    if file_exention == '.h5':
        Loader = TabularLoader.read_from_h5(args.data)
    elif file_exention == '.ulf':
        Loader = TabularLoader.read_from_flut(args.data)
    else:
        Loader = TabularLoader.read_from_csv(args.data)
    print("Done!")

    print('Normalize and split data:')  # TODO: Überlegen ob diese Argumente parsen oder besser nur im yaml
    Loader.val_split(method='random', val_size=0.2)
    Loader.test_split(methdo='random', test_size=0.05)
    print('DONE!')

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')  # TODO: Discuss with Julian if necessary
    lr_logger = LearningRateLogger()

    # # pick model
    # if args.model_name == 'MLP':
    #     model = flexMLP_pl(hparams=args)
    # elif args.model_name == 'CNN':
    #     model = flexCNN_pl(hparams=args)

    if hparams.model:
        print('Load Network and start training:')
        model = flexMLP_pl.load_from_checkpoint(hparams.model, TabularLoader=Loader)
        trainer = pl.Trainer.from_argparse_args(args, resume_from_checkpoint=hparams.model,
                                                early_stop_callback=early_stop_callback, callbacks=[lr_logger])
        trainer.fit(model)
    elif hparams.features and hparams.labels and hparams.hidden_layer:
        print('Construct Network and start training:')

        hparams.x_scaler = None
        hparams.y_scaler = None

        model = flexMLP_pl(hparams=hparams, TabluarLoader=Loader)
        if hparams.scheduler:
            trainer = pl.Trainer.from_argparse_args(args, early_stop_callback=early_stop_callback,
                                                    callbacks=[lr_logger])  # TODO: are callbacks saved by the checkpoint?
        else:
            trainer = pl.Trainer.from_argparse_args(args, early_stop_callback=early_stop_callback)
        trainer.fit(model)


if __name__ == '__main__':
    hparams, args = parseArguments()

    main(hparams, args)
