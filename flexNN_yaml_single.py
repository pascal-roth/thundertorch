#######################################################################################################################
# Load arguments of flexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils import models  # Models that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import loader  # Loader that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils.logger.tensorboard import TensorBoardLoggerAdjusted


def parse_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_yaml', type=str, default='flexMLP_single.yaml',
                        help='Name of yaml file to construct Neural Network')
    args = parser.parse_args()

    flexMLP_yaml = open(args.name_yaml)
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def get_model(argsModel):
    argsModel = argparse.Namespace(**argsModel)

    assert argsModel.source in ['load', 'create'], 'Decide if source is "load" or "create"! {} not a valid source'.\
        format(argsModel.source)
    assert hasattr(models, argsModel.type), '{} not an implemented model'.format(argsModel.type)

    if argsModel.source == 'load':
        model = getattr(models, argsModel.type).load_from_checkpoint(argsModel.load_model['path'])
        model.hparams_update(update_dict=argsModel.params)
    elif argsModel.source == 'create':
        model = getattr(models, argsModel.type)(hparams=argparse.Namespace(**argsModel.create_model, **argsModel.params))
    else:
        raise SyntaxError('Model neither loaded nor created! Set source value to "load" or "create".')

    return model


def get_dataLoader(argsLoader, model):
    argsLoader = argparse.Namespace(**argsLoader)

    assert hasattr(loader, argsLoader.type), '{} not an implemented loader'.format(argsLoader.type)

    dataLoader = getattr(loader, argsLoader.type).read_from_yaml(argsLoader, batch=model.hparams.batch,
                                                                 num_workers=model.hparams.num_workers)
    model.hparams_update(update_dict=dataLoader.lparams)
    return dataLoader


def train_model(model, dataLoader, argsTrainer) -> None:
    argsTrainer = argparse.Namespace(**argsTrainer)

    # create callback objects
    if hasattr(argsTrainer, 'callbacks'):
        callbacks = []
        if not isinstance(argsTrainer.callbacks, list): argsTrainer.callbacks = list(argsTrainer.callbacks)
        for i in range(len(argsTrainer.callbacks)):
            assert hasattr(pl.callbacks, argsTrainer.callbacks[i]['type']), '{} callback is not available in lightning'.\
                format(argsTrainer.callbacks[i]['type'])
            if 'params' in argsTrainer.callbacks[i]:
                callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])(**argsTrainer.callbacks[i]['params'])
            else:
                callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])
            callbacks.append(callback)
        argsTrainer.params['callbacks'] = callbacks

    # choose Logger
    if hasattr(argsTrainer, 'logger'):
        assert hasattr(pl.loggers, argsTrainer.logger), '{} logger not available in lightning'.format(argsTrainer.logger)
        argsTrainer.params['logger'] = getattr(pl.loggers, argsTrainer.logger)

    # define trainer and start training
    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**argsTrainer.params))
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())


def main(argsLoader, argsModel, argsTrainer):
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    argsLoader      - Namespace object with arguments related to create TabluarLoader object
    argsModel       - Namespace object with arguments related to load/ create LightningModule
    argsTrainer     - Namespace object with arguments realted to training of LightningModule
    """
    model = get_model(argsModel)
    dataLoader = get_dataLoader(argsLoader, model)
    train_model(model, dataLoader, argsTrainer)


if __name__ == '__main__':
    args_yaml = parse_yaml()

    assert 'DataLoader' in args_yaml, ''  # TODO: define statements
    assert 'Model' in args_yaml, ''
    assert 'Model' in args_yaml, ''

    argsLoader = args_yaml['DataLoader']
    argsModel = args_yaml['Model']
    argsTrainer = args_yaml['Trainer']

    main(argsLoader, argsModel, argsTrainer)