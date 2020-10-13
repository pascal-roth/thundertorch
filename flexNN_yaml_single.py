#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import argparse
import logging
import glob
import inspect
import os
import pytorch_lightning as pl

from stfs_pytoolbox.ML_Utils import models  # Models that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import loader  # Loader that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import logger  # Logger that are defined in __all__ in the __init__ file


def parse_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_yaml', type=str, default='input_LightningFlexMLP_single.yaml',
                        help='Name of yaml file to construct Neural Network')
    args = parser.parse_args()

    flexMLP_yaml = open(args.name_yaml)
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def get_model(argsModel):
    argsModel = argparse.Namespace(**argsModel)

    assert all(hasattr(argsModel, attr) for attr in ['type', 'source']), 'Model requires "type" and "source" ' \
                                                                         'definition! Please follow the template: \n{}'.\
        format(models.LightningTemplateModel.yaml_template(['Model']))
    assert hasattr(models, argsModel.type), '"{}" not an implemented model! Possible options are: "LightningFlexMLP",' \
                                            ' "LightningFLexCNN" (more will come soon)'.format(argsModel.type)
    if not hasattr(argsModel, 'params'):
        logging.warning('Parameter dict not defined! Default values will be taken. Structure of the params dict is as '
                        'follows: \n{}'.format(getattr(models, argsModel.type).yaml_template(['Model', 'params'])))

    if argsModel.source == 'load':
        assert hasattr(argsModel, 'load_model'), 'Parameter dict: "load_model" is required if source=load. ' \
                                                 '"load_model" has following structure: \n{}'.\
            format(getattr(models, argsModel.type).yaml_template(['Model', 'load_model']))
        assert 'path' in argsModel.load_model, 'Definition of path to load model is missing!'

        model = getattr(models, argsModel.type).load_from_checkpoint(argsModel.load_model['path'])
        model.hparams_update(update_dict=argsModel.params)

    elif argsModel.source == 'create':
        assert hasattr(argsModel, 'create_model'), 'Parameter dict: "create_model" is required if source=create. ' \
                                                   '"create_model" has following structure: \n{}'. \
            format(getattr(models, argsModel.type).yaml_template(['Model', 'create_model']))

        model = getattr(models, argsModel.type)(hparams=argparse.Namespace(**argsModel.create_model, **argsModel.params))

    else:
        raise ValueError('Model neither loaded nor created! Set source value to "load" or "create"."{}" not a valid '
                         'source'.format(argsModel.source))

    return model


def get_dataLoader(argsLoader, model):
    argsLoader = argparse.Namespace(**argsLoader)

    assert hasattr(argsLoader, 'type'), 'DataLoader requires "type" definition! Please follow the template: \n{}'. \
        format(loader.DataLoaderTemplate.yaml_template(['DataLoader']))
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

            assert 'type' in argsTrainer.callbacks[i], 'Each callback requires definition of the "type". Please follow' \
                                                       'the structure defined as follows: \n{}'.\
                format(trainer_yml_template(['Trainer', 'callbacks']))
            assert hasattr(pl.callbacks, argsTrainer.callbacks[i]['type']), '{} callback is not available in lightning'.\
                format(argsTrainer.callbacks[i]['type'])

            if 'params' in argsTrainer.callbacks[i]:
                callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])(**argsTrainer.callbacks[i]['params'])
            else:
                callback = getattr(pl.callbacks, argsTrainer.callbacks[i]['type'])

            callbacks.append(callback)

        argsTrainer.params['callbacks'] = callbacks

    else:
        logging.info('No callbacks implemented')

    # create logger_fn objects
    if hasattr(argsTrainer, 'logger'):
        loggers = []
        if not isinstance(argsTrainer.logger, list): argsTrainer.logger = list(argsTrainer.logger)

        for i in range(len(argsTrainer.logger)):

            assert 'type' in argsTrainer.logger[i], 'Each logger_fn requires definition of the "type". Please follow' \
                                                    'the structure defined as follows: \n{}'.\
                format(trainer_yml_template(['Trainer', 'logger_fn']))

            if argsTrainer.logger[i]['type'] == 'comet-ml':
                from pytorch_lightning.loggers.comet import CometLogger
                logger_fn = CometLogger(**argsTrainer.logger[i]['params'])
            elif argsTrainer.logger[i]['type'] == 'tensorboard':
                logger_fn = logger.TensorBoardLoggerAdjusted
            else:
                raise ValueError('Selected logger not implemented!')

            loggers.append(logger_fn)

        argsTrainer.params['logger'] = loggers

    else:
        argsTrainer.params['logger'] = False
        logging.info('No logger selected')

    # define trainer and start training
    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**argsTrainer.params))
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())


def trainer_yml_template(key_list):
    template = {'Trainer': {'params': {'gpus': 'int', 'max_epochs': 'int', 'profiler': 'bool'},
                            'callbacks': [{'type': 'EarlyStopping',
                                           'params': {'monitor': 'val_loss', 'patience': 'int', 'mode': 'min'}},
                                          {'type': 'ModelCheckpoint',
                                           'params': {'filepath': 'None', 'save_top_k': 'int'}},
                                          {'type': 'lr_logger'}],
                            'logger': [{'type': 'Comet-ml',
                                        'params': {'api_key': 'personal_comet_api', 'project_name': 'str',
                                                   'workspace': 'personal_comet_workspace', 'experiment_name': 'name'}},
                                       {'type': 'Tensorboard'}]}}

    for i, key in enumerate(key_list):
        template = template.get(key)

    return yaml.dump(template, sort_keys=False)


def main(args_yaml):
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    args_yaml       - Dict with all input arguments
    """

    assert 'DataLoader' in args_yaml, 'Training a model requires some data which is packed inside a DataLoader! ' \
                                      'Definiton of the DataLoader type and the corresponding parameters is missing. ' \
                                      'DataLoaders can be found under stfs_pytoolbox/ML_utils/loader. The tempolate ' \
                                      'yml structure for a DataLoader is defined as follows: \n{}'.\
        format(loader.DataLoaderTemplate.yaml_template([]))

    assert 'Model' in args_yaml, 'Neural Network Model definition is missing! Possible models are {}. The template ' \
                                 'yml structure for the Models is defined as follows: \n{}'.\
        format(glob.glob(os.path.dirname(inspect.getfile(models)) + '/Lightning*'),
               models.LightningTemplateModel.yaml_template([]))

    assert 'Trainer' in args_yaml, 'No Trainer of the Network defined! The trainer is responsible for automating ' \
                                   'network training, tesing and saving. A detailed description of the possible ' \
                                   'parameters is given at: https://pytorch-lightning.readthedocs.io/en/latest/' \
                                   'trainer.html. The yml structure to include a trainer is as follows: \n{}'.\
        format(trainer_yml_template([]))

    argsLoader = args_yaml['DataLoader']
    argsModel = args_yaml['Model']
    argsTrainer = args_yaml['Trainer']

    model = get_model(argsModel)
    dataLoader = get_dataLoader(argsLoader, model)
    train_model(model, dataLoader, argsTrainer)


if __name__ == '__main__':
    args_yaml = parse_yaml()
    main(args_yaml)
