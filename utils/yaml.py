#######################################################################################################################
# Util functions to process yaml files
#######################################################################################################################

# import packages
import yaml
import glob
import inspect
import os
import pytorch_lightning as pl
from functools import reduce
import operator

from stfs_pytoolbox.ML_Utils import _logger
from stfs_pytoolbox.ML_Utils import models  # Models that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils import loader  # Loader that are defined in __all__ in the __init__ file
from stfs_pytoolbox.ML_Utils.utils.option_class import OptionClass
from stfs_pytoolbox.ML_Utils import _modules_models, _modules_callbacks, _modules_loader


def parse_yaml(yaml_path) -> dict:
    """
    Parse yaml file
    """
    flexMLP_yaml = open(yaml_path)
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def check_yaml_version(args_yaml: dict) -> None:  # TODO: assert error if yaml file changed with a new version
    # stfs_pytoolbox.__version__
    pass


def check_args(argsModel: dict, argsLoader: dict, argsTrainer: dict) -> None:
    # transform to namespace objects
    check_argsModel(argsModel)
    check_argsLoader(argsLoader)
    check_argsTrainer(argsTrainer)


def check_argsModel(argsModel: dict) -> None:
    """
    Control Model arguments regarding included keys, dtypes of the keys, mutually_exclusive relations and whether
    the intended attr of a function exists

    Parameters
    ----------
    argsModel       - Dict including the model arguments of a yaml file
    """
    options = {'Model': OptionClass(template=models.LightningModelTemplate.yaml_template(['Model']))}
    options['Model'].add_key('type', dtype=str, required=True, attr_of=_modules_models)
    options['Model'].add_key('load_model', dtype=dict, mutually_exclusive=['create_model'])
    options['Model'].add_key('create_model', dtype=dict, mutually_exclusive=['load_model'], param_dict=True)
    options['Model'].add_key('params', dtype=dict, param_dict=True)

    options['load_model'] = OptionClass(template=models.LightningModelTemplate.yaml_template(['Model', 'load_model']))
    options['load_model'].add_key('path', dtype=str, required=True)

    OptionClass.checker(input_dict={'Model': argsModel}, option_classes=options)

    # warn if no model params defined
    if 'params' not in argsModel:
        _logger.warning('Parameter dict not defined! Default values will be taken. Structure of the params dict is as '
                        'follows: \n{}'.format(getattr(models, argsModel.type).yaml_template(['Model', 'params'])))

    # check model source
    if 'load_model' not in argsModel and 'create_model' not in argsModel:
        raise KeyError('Definition of load or create model dict necessary!')


def check_argsLoader(argsLoader: dict) -> None:
    """
    Control DataLoader arguments regarding included keys, dtypes of the keys, mutually_exclusive relations and whether
    the intended attr of a function exists

    Parameters
    ----------
    argsLoader      - Dict including the DataLoader arguments of a yaml file
    """
    options = {'DataLoader': OptionClass(template=loader.DataLoaderTemplate.yaml_template(['DataLoader']))}
    options['DataLoader'].add_key('type', dtype=str, required=True, attr_of=_modules_loader)
    options['DataLoader'].add_key('load_DataLoader', dtype=dict, mutually_exclusive=['create_DataLoader'], param_dict=True)
    options['DataLoader'].add_key('create_DataLoader', dtype=dict, mutually_exclusive=['load_DataLoader'], param_dict=True)

    OptionClass.checker(input_dict={'DataLoader': argsLoader}, option_classes=options)


def check_argsTrainer(argsTrainer: dict) -> None:
    """
    Control Trainer arguments regarding included keys, dtypes of the keys, mutually_exclusive relations and whether
    the intended attr of a function exists

    Parameters
    ----------
    argsTrainer     - Dict including the trainer arguments of a yaml file
    """
    options = {'Trainer': OptionClass(template=trainer_yml_template(['Trainer']))}
    options['Trainer'].add_key('params', dtype=dict, param_dict=True)
    options['Trainer'].add_key('callbacks', dtype=[dict, list])
    options['Trainer'].add_key('logger', dtype=[dict, list])

    options['callbacks'] = OptionClass(template=trainer_yml_template(['Trainer', 'callbacks']))
    options['callbacks'].add_key('type', dtype=str, required=True, attr_of=_modules_callbacks)
    options['callbacks'].add_key('params', dtype=dict, param_dict=True)

    options['logger'] = OptionClass(template=trainer_yml_template(['Trainer', 'logger']))
    options['logger'].add_key('type', dtype=str, required=True)
    options['logger'].add_key('params', dtype=dict, param_dict=True)

    OptionClass.checker(input_dict={'Trainer': argsTrainer}, option_classes=options)

    if all(elem in argsTrainer['params'] for elem in ['gpus', 'profiler']) and argsTrainer['params']['gpus'] != 0 and \
            argsTrainer['params']['profiler'] is True:
        raise KeyError('In multi GPU training, profiler cannot be active!')


def check_yaml_structure(args_yaml: dict) -> None:
    """
    Control if yaml file consist out of DataLoader, Model and Trainer argument dicts

    Parameters
    ----------
    args_yaml       - parsed yaml dict
    """
    assert 'DataLoader' in args_yaml, 'Training a model requires some data which is packed inside a DataLoader! ' \
                                      'Definiton of the DataLoader type and the corresponding parameters is missing. ' \
                                      'DataLoaders can be found under stfs_pytoolbox/ML_utils/loader. The tempolate ' \
                                      'yml structure for a DataLoader is defined as follows: \n{}'.\
        format(loader.DataLoaderTemplate.yaml_template([]))

    assert 'Model' in args_yaml, 'Neural Network Model definition is missing! Possible models are {}. The template ' \
                                 'yml structure for the Models is defined as follows: \n{}'.\
        format(glob.glob(os.path.dirname(inspect.getfile(models)) + '/Lightning*'),
               models.LightningModelTemplate.yaml_template([]))

    assert 'Trainer' in args_yaml, 'No Trainer of the Network defined! The trainer is responsible for automating ' \
                                   'network training, tesing and saving. A detailed description of the possible ' \
                                   'parameters is given at: https://pytorch-lightning.readthedocs.io/en/latest/' \
                                   'trainer.html. The yml structure to include a trainer is as follows: \n{}'.\
        format(trainer_yml_template([]))


def trainer_yml_template(key_list: list) -> dict:
    """
    Trainer yaml template
    """
    template = {'Trainer': {'params': {'gpus': 'int', 'max_epochs': 'int', 'profiler': 'bool'},
                            'callbacks': [{'type': 'EarlyStopping',
                                           'params': {'monitor': 'val_loss', 'patience': 'int', 'mode': 'min'}},
                                          {'type': 'LearningRateLogger'},
                                          {'type': 'Checkpointing',
                                           'params': {'filepath': 'None', 'save_top_k': 'int'}},
                                          {'type': 'lr_logger'}],
                            'logger': [{'type': 'Comet-ml',
                                        'params': {'api_key': 'personal_comet_api', 'project_name': 'str',
                                                   'workspace': 'personal_comet_workspace', 'experiment_name': 'name'}},
                                       {'type': 'Tensorboard'}]}}

    for i, key in enumerate(key_list):
        template = template.get(key)

    return yaml.dump(template, sort_keys=False)


def replace_keys(dictMultiModel: dict, dictSingleModel: dict) -> dict:
    """
    Take keys given in the definition of a Model in the MultiModel file and either add them to the template SingleModel
    file or replace the corresponding key value in it

    Parameters
    ----------
    dictMultiModel      - dict including all keys that should be add/ changed in template
    dictSingleModel     - template model dict

    Returns
    -------
    dictRunModel        - adjusted model dict
    """

    # get, set and del keys in nested dict structure
    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root, items, value):
        """Set a value in a nested object in root by item sequence."""
        get_by_path(root, items[:-1])[items[-1]] = value

    def del_by_path(root, items):
        """Delete a key-value in a nested object in root by item sequence."""
        del get_by_path(root, items[:-1])[items[-1]]

    def recursion_search(document: dict, key_list: list, dictModel: dict):
        """
        Recursive function to add/ replace key in a nested dict

        Parameters
        ----------
        document        - dict with the keys that should be add
        key_list        - key path in the nested dict to the final key
        dictModel       - model dict where the keys are changed/ added

        Returns
        -------
        dictModel       - model dict where the keys are changed/ added
        key_list        - key path in the nested dict to the final key
        """
        if isinstance(document, dict):
            for key, value in document.items():
                key_list.append(key)
                dictModel, key_list = recursion_search(document=value, key_list=key_list,
                                                       dictModel=dictModel)
                key_list = key_list[:-1]

        elif isinstance(document, list) and all(isinstance(elem, dict) for elem in document):
            for list_dict in document:
                dictSingleModel_list_dict = get_by_path(dictModel, key_list)
                dictSingleModel_list_dict_nbr = next((i for i, item in enumerate(dictSingleModel_list_dict)
                                                      if item["type"] == list_dict['type']))
                key_list.extend([dictSingleModel_list_dict_nbr, 'params'])
                dictModel, key_list = recursion_search(document=list_dict['params'], key_list=key_list,
                                                       dictModel=dictModel)
                key_list = key_list[:-2]

        else:
            set_by_path(dictModel, key_list, document)

        return dictModel, key_list

    dictRunModel, _ = recursion_search(document=dictMultiModel, key_list=list([]), dictModel=dictSingleModel)

    return dictRunModel
