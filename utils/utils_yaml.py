#######################################################################################################################
# Util functions to process yaml files
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
from stfs_pytoolbox.ML_Utils import callbacks  # Callbacks that are defined in __all__ in the __init__ file


def parse_yaml() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_yaml', type=str, default='input_LightningFlexMLP_single.yaml',
                        help='Name of yaml file to construct Neural Network')
    args = parser.parse_args()

    flexMLP_yaml = open(args.name_yaml)
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def check_args(argsModel, argsLoader, argsTrainer) -> None:
    # transform to namespace objects
    check_argsModel(argsModel)
    check_argsLoader(argsLoader)
    check_argsTrainer(argsTrainer)


def check_argsModel(argsModel) -> None:
    if isinstance(argsModel, dict):
        argsModel = argparse.Namespace(**argsModel)

    # check model type
    assert hasattr(argsModel, 'type'), 'Model requires "type" definition! Please follow the template: \n{}'. \
        format(models.LightningTemplateModel.yaml_template(['Model']))

    assert hasattr(models, argsModel.type), '"{}" not an implemented model! Possible options are: "LightningFlexMLP",' \
                                            ' "LightningFLexCNN" (more will come soon)'.format(argsModel.type)

    # warn if no model params defined
    if not hasattr(argsModel, 'params'):
        logging.warning('Parameter dict not defined! Default values will be taken. Structure of the params dict is as '
                        'follows: \n{}'.format(getattr(models, argsModel.type).yaml_template(['Model', 'params'])))

    # check model source
    if hasattr(argsModel, 'load_model') and hasattr(argsModel, 'create_model'):
        assert hasattr(argsModel, 'source'), 'Both dicts "load_model" and "create_model" are defined. Either define ' \
                                             'source flag or remove unintended dict'
        assert argsModel.source in ['load', 'create'], 'Source flag is either "load" or "create". "{}" not valid!'. \
            format(argsModel.source)

    elif hasattr(argsModel, 'source'):
        if argsModel.source == 'load':
            assert hasattr(argsModel, 'load_model'), 'Parameter dict: "load_model" is required if source=load. ' \
                                                     '"load_model" has following structure: \n{}'. \
                format(getattr(models, argsModel.type).yaml_template(['Model', 'load_model']))

        elif argsModel.source == 'create':
            assert hasattr(argsModel, 'create_model'), 'Parameter dict: "create_model" is required if source=create. ' \
                                                       '"create_model" has following structure: \n{}'. \
                format(getattr(models, argsModel.type).yaml_template(['Model', 'create_model']))

        else:
            raise ValueError(
                'Model neither loaded nor created! Set source value to "load" or "create" and include corresponding '
                'dict."{}" not a valid source'.format(argsModel.source))

    elif not hasattr(argsModel, 'load_model') and hasattr(argsModel, 'create_model'):
        raise KeyError('Definition of load or create model dict necessary!')

    if hasattr(argsModel, 'load_model'):
        assert 'path' in argsModel.load_model, 'Definition of path to load model is missing!'


def check_argsLoader(argsLoader) -> None:
    if isinstance(argsLoader, dict):
        argsLoader = argparse.Namespace(**argsLoader)

    assert hasattr(argsLoader, 'type'), 'DataLoader requires "type" definition! Please follow the template: \n{}'. \
        format(loader.DataLoaderTemplate.yaml_template(['DataLoader']))
    assert hasattr(loader, argsLoader.type), '{} not an implemented loader'.format(argsLoader.type)


def check_argsTrainer(argsTrainer) -> None:
    if isinstance(argsTrainer, dict):
        argsTrainer = argparse.Namespace(**argsTrainer)

    if hasattr(argsTrainer, 'callbacks'):
        if not isinstance(argsTrainer.callbacks, list): argsTrainer.callbacks = list(argsTrainer.callbacks)

        assert all('type' in callback for callback in argsTrainer.callbacks), \
            'Each callback requires definition of the "type". Please follow the structure defined as follows: \n{}'. \
            format(trainer_yml_template(['Trainer', 'callbacks']))

        assert all((hasattr(pl.callbacks, callback['type']) or hasattr(callbacks, callback['type'])) for callback in argsTrainer.callbacks),\
            'Callback not available in lightning and not self-implemented'

    if hasattr(argsTrainer, 'logger'):
        if not isinstance(argsTrainer.logger, list): argsTrainer.logger = list(argsTrainer.logger)

        assert all('type' in logger for logger in argsTrainer.logger), \
            'Each logger_fn requires definition of the "type". Please follow the structure defined as follows: \n{}'. \
            format(trainer_yml_template(['Trainer', 'logger_fn']))


def check_yaml_structure(args_yaml: dict) -> None:
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


def trainer_yml_template(key_list) -> dict:
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


def replace_keys(dictionary, yamlTemplate):
    def recursion(document, key_list, yamlTemplate):
        if isinstance(document, dict):
            for key, value in document.items():
                key_list.append(key)
                yamlTemplate, key_list = recursion(document=value, key_list=key_list, yamlTemplate=yamlTemplate)
                key_list = key_list[:-1]
        else:

            if len(key_list) == 2:
                if all(key not in ['params', 'val_params', 'test_params'] for key in key_list):
                    assert key_list[1] in yamlTemplate[key_list[0]], \
                        'Key {} not included in yaml_template'.format(key_list)
                yamlTemplate[key_list[0]].update({key_list[1]: document})

            elif len(key_list) == 3:
                if all(key not in ['params', 'val_params', 'test_params'] for key in key_list):
                    assert key_list[2] in yamlTemplate[key_list[0]][key_list[1]], \
                        'Key {} not included in yaml_template'.format(key_list)
                yamlTemplate[key_list[0]][key_list[1]].update({key_list[2]: document})

            elif len(key_list) == 4:
                if all(key not in ['params', 'val_params', 'test_params'] for key in key_list):
                    assert key_list[3] in yamlTemplate[key_list[0]][key_list[1]][key_list[2]], \
                        'Key {} not included in yaml_template'.format(key_list)
                yamlTemplate[key_list[0]][key_list[1]][key_list[2]].update({key_list[3]: document})

            elif len(key_list) == 5:
                if all(key not in ['params', 'val_params', 'test_params'] for key in key_list):
                    assert key_list[4] in yamlTemplate[key_list[0]][key_list[1]][key_list[2]][key_list[3]], \
                        'Key {} not included in yaml_template'.format(key_list)
                yamlTemplate[key_list[0]][key_list[1]][key_list[2]][key_list[3]].update({key_list[4]: document})

            else:
                raise IndexError('Depth of multi yaml key {} is out of range of template keys'.format(key_list))

        return yamlTemplate, key_list

    yamlTemplate, _ = recursion(document=dictionary, key_list=list([]), yamlTemplate=yamlTemplate)

    return yamlTemplate
