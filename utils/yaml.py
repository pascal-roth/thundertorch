#######################################################################################################################
# Util functions to process yaml files
#######################################################################################################################

# import packages
import yaml
import glob
import inspect
import os
from functools import reduce
import operator
import torch
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


# argument checking ###################################################################################################
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


def check_argsConfig_single(argsConfig: dict) -> None:
    """
    Control Config arguments regarding included keys, dtypes of the keys, mutually_exclusive relations for the
    single model yaml

    Parameters
    ----------
    argsConfig    - Dict including the Config arguments of the single model yaml
    """
    options = {'config': OptionClass(template=config_yml_template([]))}
    options['config'].add_key('source_files', dtype=str)
    options['config'].add_key('reproducibility', dtype=bool)

    OptionClass.checker(input_dict={'config': argsConfig}, option_classes=options)


def check_argsConfig_multi(argsConfig: dict) -> None:
    """
    Control Config arguments regarding included keys, dtypes of the keys, mutually_exclusive relations for the
    multi model yaml

    Parameters
    ----------
    argsConfig    - Dict including the Config arguments of the multi model yaml
    """
    options = {'config': OptionClass(template=multimodel_training_yml_template(['config']))}
    options['config'].add_key('nbr_processes', dtype=int)
    options['config'].add_key('GPU_per_model', dtype=int, mutually_exclusive=['CPU_per_model'])
    options['config'].add_key('CPU_per_model', dtype=int, mutually_exclusive=['GPU_per_model'])
    options['config'].add_key('model_run', dtype=list)

    OptionClass.checker(input_dict={'config': argsConfig}, option_classes=options)


# structure checking ###################################################################################################
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


# yaml template for trainer and config arguments, as well as the MultiModel Training ##################################
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


def config_yml_template(key_list: list) -> dict:
    """
    Config template for single model yaml
    """
    template = {'config': {'source_files': 'individual module (individual function, model, loader, ... has to be '
                                           'mentioned in __all__ in the __init__ of the module, so that it can be '
                                           'loaded)',
                           'reproducibility': 'True'}}

    for i, key in enumerate(key_list):
        template = template.get(key)

    return yaml.dump(template, sort_keys=False)


def multimodel_training_yml_template(key_list: list, template: str = 'path.yaml (required!)') -> dict:
    """
    Template for Multi-Model Training
    """

    template = {'config': {'###INFO###': '"CPU_per_model" and "GPU_per_model" mutually exclusive',
                           'CPU_per_model': 'int', 'GPU_per_model': 'int',
                           'nbr_processes': 'int', 'model_run': ['Model001', 'Model002', 'model_name_3', '...']},
                'Model001': {'Template': template,
                             '###INFO###': 'After template defintion, keys of the template can be changed or new '
                                           'keys added. The key structure has to be the same. Here an example is given',
                             'DataLoader': {'create_DataLoader': {'raw_data_path': 'different_path.csv',
                                                                  'features': ['feature_1', 'feature_2'],
                                                                  'labels': ['label_1', 'label_2']}},
                             'Model': {'create_model': {'n_inp': 'int', 'n_out': 'int', 'hidden_layer': ['int', 'int']}},
                             'Trainer': {'params': {'max_epochs': 'int'},
                                         'callbacks': [{'type': 'Checkpointing', 'params': {'filepath': 'path'}}]}},
                'Model002': {'Template': template,
                             'DataLoader': {'create_DataLoader': {'raw_data_path': 'different_path.csv',
                                                                  'features': ['feature_1', 'feature_2'],
                                                                  'labels': ['label_1', 'label_2']}},
                             'Model': {'create_model': {'n_inp': 'int', 'n_out': 'int', 'hidden_layer': ['int', 'int']},
                                       'params': {'optimizer': {'type': 'SGD', 'params': {'lr': 0.001}}}},
                             'Trainer': {'params': {'max_epochs': 'int'}}}}

    for i, key in enumerate(key_list):
        template = template.get(key)

    return yaml.dump(template, sort_keys=False)


# configure MultiModel Training #######################################################################################
def config_multi(argsMulti: dict) -> tuple:
    """
    configure MultiModel Training
    """
    argsModels, argsConfig = get_argsModel(argsMulti)
    nbr_processes, list_gpu = get_num_processes(argsConfig, argsModels)
    model_dicts = get_argsDict(argsModels)
    return nbr_processes, list_gpu, model_dicts


def get_argsModel(argsMulti):
    """
    Select the defined models which should be trained
    """
    # check if config dict in yaml
    if 'config' in argsMulti:
        argsConfig = argsMulti.pop('config')
        check_argsConfig_multi(argsConfig)
        _logger.debug('Config file included and controlled')
    else:
        _logger.debug('No Config file included in MultiModel Training yaml!')
        argsConfig = []

    # filter for models defined in Model_Run list
    if 'model_run' in argsConfig:
        model_run_list = argsConfig.pop('model_run')
        assert all(elem in argsMulti for elem in model_run_list), 'Model name included in "model_run" not found!'
        argsModels = {model_key: argsMulti[model_key] for model_key in model_run_list}
        assert len(argsModels) != 0, 'No models defined in "input_MultiModelTraining.yaml"!'
    else:
        argsModels = argsMulti
        _logger.debug('No Models excluded! All models selected for training!')

    return argsModels, argsConfig


def get_argsDict(argsModels):
    """
    Get the defined keys in the MultiModel yaml and compare it to the template. Final keys are given back for training
    """
    model_list = list(argsModels)
    model_dicts = []
    for ModelName in model_list:
        argsModel = argsModels[ModelName]

        assert 'Template' in argsModel, 'Definition of a Template necessary to change model keys!'
        yamlTemplate_location = argsModel.pop('Template')

        with open(yamlTemplate_location) as yaml_file:
            yamlTemplate = yaml.load(yaml_file, Loader=yaml.FullLoader)
            yamlModelRun = replace_keys(argsModel, yamlTemplate)

        model_dicts.append(yamlModelRun)

    return model_dicts


def get_num_processes(argsConfig: dict, argsModels: dict) -> tuple:
    """
    Identify available resources and determine the number of processes which are executed simultaneously
    """
    # get nbr of parallel processes for multiprocessing
    nbr_cpu = os.cpu_count()  # nbr of available CPUs
    nbr_gpu = torch.cuda.device_count()  # nbr of available GPUs
    gpu_per_process = 0

    if 'GPU_per_model' in argsConfig:
        assert nbr_gpu != 0, 'GPU per process defined. but NO GPU available!'
        gpu_per_process = argsConfig.pop('GPU_per_model')

        nbr_processes = int((nbr_gpu - (nbr_gpu % gpu_per_process)) / gpu_per_process)
        assert nbr_processes != 0, f'Not enough GPUs! Model should be trained with {gpu_per_process} GPU(s), but' \
                                 f'available are only {nbr_gpu} GPU(s)'

        _logger.debug(f'{nbr_processes} processes can be executed with {gpu_per_process} GPU(s) per process')

    elif 'CPU_per_model' in argsConfig:
        cpu_per_process = argsConfig.pop('CPU_per_model')

        nbr_processes = int((nbr_cpu - (nbr_cpu % cpu_per_process)) / cpu_per_process)
        assert nbr_processes != 0, f'Not enough CPUs! Model should be trained with {cpu_per_process} CPU(s), but' \
                                 f'available are only {nbr_cpu} CPU(s)'

        _logger.debug(f'{nbr_processes} processes can be executed with {cpu_per_process} CPU(s) per process')

    else:
        _logger.debug('Neither "GPU_per_model" nor "CPU_per_model" defined, default setting is selected')
        if nbr_gpu != 0:
            gpu_per_process = 1
            nbr_processes = nbr_gpu
            _logger.debug(f'{nbr_processes} processes can be executed with {gpu_per_process} GPU(s) per process')
        else:
            cpu_per_process = 1
            nbr_processes = nbr_cpu
            _logger.debug(f'{nbr_processes} processes can be executed with {cpu_per_process} CPU(s) per process')

    if 'nbr_processes' in argsConfig:
        _logger.debug(f'Config nbr_processes {argsConfig["nbr_processes"]} is compared with maximal possible number of '
                      f'processes {nbr_processes}, minimum is selected')
        nbr_processes = min(nbr_processes, argsConfig['nbr_processes'])

    nbr_processes = min(len(argsModels), nbr_processes)

    if gpu_per_process != 0 and nbr_gpu != 0:
        list_gpu = []
        gpu_available = list(range(0, nbr_gpu))
        for i in range(nbr_processes):
            list_gpu.append(gpu_available[0:gpu_per_process])
            del gpu_available[0:gpu_per_process]
    else:
        list_gpu = [0 for x in range(nbr_processes)]

    return nbr_processes, list_gpu


# function used in MultiModel Training to replace keys in the template  ###############################################
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
            try:
                set_by_path(dictModel, key_list, document)
            except KeyError:
                raise KeyError(f'The given key list {key_list[:-1]} towards the key [{key_list[-1]}] which should be '
                               f'added or changed is incorrect. Keep attention that only the last key can be edited, '
                               f'all keys prior have to be included in the template \n'
                               f'e.g. for a key_list = ["DataLoader", "load_DataLoader", "path"] the template must '
                               f'include the "load_DataLoader dict')

        return dictModel, key_list

    dictRunModel, _ = recursion_search(document=dictMultiModel, key_list=list([]), dictModel=dictSingleModel)

    return dictRunModel

