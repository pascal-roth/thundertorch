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
from thunder_torch import _logger
from thunder_torch import models  # Models that are defined in __all__ in the __init__ file
from thunder_torch import loader  # Loader that are defined in __all__ in the __init__ file
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_models, _modules_callbacks, _modules_loader


def parse_yaml(yaml_path: str, low_key: bool = True) -> dict:
    """
    Parse yaml file and lower case all keys
    """
    flexMLP_yaml = open(yaml_path)
    yaml_dict = yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)
    if low_key:
        yaml_dict = lower_keys(yaml_dict)
    return yaml_dict


def lower_keys(dict_file: dict) -> dict:

    def recursion(dict_file_rec: dict):
        dict_file_rec = dict((k.lower(), v) for k, v in dict_file_rec.items())

        for key, value in dict_file_rec.items():
            if key == 'split_data':
                pass
            elif isinstance(value, dict):
                dict_file_rec[key] = recursion(value)
            elif isinstance(value, list) and all(isinstance(elem, dict) for elem in value):
                for i, list_dict in enumerate(value):
                    value[i] = recursion(list_dict)
                dict_file_rec[key] = value

        return dict_file_rec

    dict_file = recursion(dict_file)
    return dict_file


def check_yaml_version(args_yaml: dict) -> None:  # TODO: assert error if yaml file changed with a new version
    # thunder_torch.__version__
    pass


# argument checking ###################################################################################################
def check_args(argsYaml: dict) -> None:
    check_argsModel(argsYaml['model'])
    check_argsLoader(argsYaml['dataloader'])
    check_argsTrainer(argsYaml['trainer'])


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
    options['DataLoader'].add_key('load_dataloader', dtype=dict, mutually_exclusive=['create_dataloader'], param_dict=True)
    options['DataLoader'].add_key('create_dataloader', dtype=dict, mutually_exclusive=['load_dataloader'], param_dict=True)

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
    options['Trainer'].add_key('params', dtype=dict, param_dict=True, required=True)
    options['Trainer'].add_key('callbacks', dtype=[dict, list])
    options['Trainer'].add_key('logger', dtype=[dict, list])

    options['callbacks'] = OptionClass(template=trainer_yml_template(['Trainer', 'callbacks']))
    options['callbacks'].add_key('type', dtype=str, required=True, attr_of=_modules_callbacks)
    options['callbacks'].add_key('params', dtype=dict, param_dict=True)

    options['logger'] = OptionClass(template=trainer_yml_template(['Trainer', 'logger']))
    options['logger'].add_key('type', dtype=str, required=True)
    options['logger'].add_key('params', dtype=dict, param_dict=True, required=True)

    OptionClass.checker(input_dict={'Trainer': argsTrainer}, option_classes=options)

    if all(elem in argsTrainer['params'] for elem in ['gpus', 'profiler']) and argsTrainer['params']['gpus'] > 1 and \
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
    options['config'].add_key('deterministic', dtype=bool)

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
    options['config'].add_key('gpu_per_model', dtype=int, mutually_exclusive=['cpu_per_model'])
    options['config'].add_key('cpu_per_model', dtype=int, mutually_exclusive=['gpu_per_model'])
    options['config'].add_key('model_run', dtype=[list, str])

    OptionClass.checker(input_dict={'config': argsConfig}, option_classes=options)


# structure checking ###################################################################################################
def check_yaml_structure(args_yaml: dict) -> None:
    """
    Control if yaml file consist out of DataLoader, Model and Trainer argument dicts

    Parameters
    ----------
    args_yaml       - parsed yaml dict
    """
    assert 'dataloader' in args_yaml, 'Training a model requires some data which is packed inside a DataLoader! ' \
                                      'Definiton of the DataLoader type and the corresponding parameters is missing. ' \
                                      'DataLoaders can be found under thunder_torch/loader. The tempolate ' \
                                      'yml structure for a DataLoader is defined as follows: \n{}'.\
        format(loader.DataLoaderTemplate.yaml_template([]))

    assert 'model' in args_yaml, 'Neural Network Model definition is missing! Possible models are {}. The template ' \
                                 'yml structure for the Models is defined as follows: \n{}'.\
        format(glob.glob(os.path.dirname(inspect.getfile(models)) + '/Lightning*'),
               models.LightningModelTemplate.yaml_template([]))

    assert 'trainer' in args_yaml, 'No Trainer of the Network defined! The trainer is responsible for automating ' \
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
                                       {'type': 'Tensorboard',
                                        'params': {'save_dir': 'path', 'name': 'log_name',
                                                   'version': 'experiment version'}}]}}

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
                           'deterministic': 'True'}}

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


def get_argsModel(argsMulti: dict) -> tuple:
    """
    Select the defined models which should be trained
    """
    # check if config dict in yaml
    if 'config' in argsMulti or 'Config' in argsMulti:
        try:
            argsConfig = argsMulti.pop('config')
        except KeyError:
            argsConfig = argsMulti.pop('Config')
        argsConfig = lower_keys(argsConfig)
        check_argsConfig_multi(argsConfig)
        _logger.debug('Config file included and controlled')
        if not argsConfig: _logger.warning('Defined config tree is empty!')
    else:
        _logger.debug('No Config file included in MultiModel Training yaml!')
        argsConfig = []

    # filter for models defined in Model_Run list
    if 'model_run' in argsConfig:
        model_run_list = argsConfig.pop('model_run')

        if isinstance(model_run_list, str): model_run_list = [model_run_list]

        assert all(elem in argsMulti for elem in model_run_list), f'Model name included in "model_run": ' \
                                                                  f'{model_run_list} not found!'
        argsModels = {model_key: argsMulti[model_key] for model_key in model_run_list}
        assert len(argsModels) != 0, 'No models defined in "input_MultiModelTraining.yaml"!'
    else:
        argsModels = argsMulti
        _logger.debug('No Models excluded! All models selected for training!')

    return argsModels, argsConfig


def get_argsDict(argsModels: dict) -> list:
    """
    Get the defined keys in the MultiModel yaml and compare it to the template. Final keys are given back for training
    """
    model_list = list(argsModels)
    model_dicts = []
    for ModelName in model_list:
        argsModel = argsModels[ModelName]
        argsModel = lower_keys(argsModel)
        argsModel = replace_expression(argsModel, ModelName)

        assert 'template' in argsModel, 'Definition of a Template necessary to change model keys!'
        yamlTemplate_location = argsModel.pop('template')

        yamlTemplate = parse_yaml(yamlTemplate_location)
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

    if 'gpu_per_model' in argsConfig:
        assert nbr_gpu != 0, 'GPU per process defined. but NO GPU available!'
        gpu_per_process = argsConfig.pop('gpu_per_model')

        nbr_processes = int((nbr_gpu - (nbr_gpu % gpu_per_process)) / gpu_per_process)
        assert nbr_processes != 0, f'Not enough GPUs! Model should be trained with {gpu_per_process} GPU(s), but' \
                                 f'available are only {nbr_gpu} GPU(s)'

        _logger.debug(f'{nbr_processes} processes can be executed with {gpu_per_process} GPU(s) per process')

    elif 'cpu_per_model' in argsConfig:
        cpu_per_process = argsConfig.pop('cpu_per_model')

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
            cpu_per_process = nbr_cpu
            nbr_processes = 1
            _logger.debug(f'{nbr_processes} processes can be executed with {cpu_per_process} CPU(s) per process')

    if 'nbr_processes' in argsConfig:
        _logger.debug(f'Config nbr_processes {argsConfig["nbr_processes"]} is compared with maximal possible or '
                      f'default number of processes {nbr_processes}, minimum is selected')
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


def replace_expression(argsModel: dict, ModelName: str, expression: str = '<model_name>') -> dict:
    """
    In a multi layer dict replace the expression "{model_name}" by the defined ModelName. Convenience feature for the
    MultiModel Training
    """
    def recursion(rec_dict: dict):
        for key, value in rec_dict.items():
            if isinstance(value, str):
                rec_dict[key] = value.replace(expression, ModelName)
            elif isinstance(value, dict):
                rec_dict[key] = recursion(value)
            elif isinstance(value, list) and all(isinstance(elem, dict) for elem in value):
                for i, list_dict in enumerate(value):
                    value[i] = recursion(list_dict)
                rec_dict[key] = value
            elif isinstance(value, list) and all(isinstance(elem, str) for elem in value):
                rec_dict[key] = list(elem.replace(expression, ModelName) for elem in value)

        return rec_dict

    argsModel = recursion(argsModel)
    return argsModel


# Header for single and multi-model YAML ##############################################################################
def header(name_model: str, name_loader: str) -> tuple:
    # construct single yaml header
    header_single = f"#  +-----------------------------+\n" \
                    f"#  |  ThunderTorch Wrapper       |\n" \
                    f"#  +-----------------------------+\n \n" \
                    f"#  YAML for Single Model Training\n" \
                    f"#  ------------------------------\n \n" \
                    f"#  The YAML contains 4 parts (Config, DataLoader, Model and Trainer) which are required for " \
                    f"the ML task. This template is fitted for: \n" \
                    f"#     Model:         {name_model}\n" \
                    f"#     DataLoader:    {name_loader}\n" \
                    f"#  After adjusting the YAML run the script 'trainFlexNN' to execute the ML task.\n" \
                    f"#  For the documentation see: https://proth.pages.rwth-aachen.de/pythonToolBox/ML_utils" \
                    f"/working_examples/working_example_yaml.html \n" \
                    f"#  ------------------------------ \n"

    # construct single yaml header
    header_multi = "#  +-----------------------------+\n" \
                   "#  |  ThunderTorch Wrapper       |\n" \
                   "#  +-----------------------------+\n \n" \
                   "#  YAML for Multi Model Training \n" \
                   "#  ------------------------------\n \n" \
                   "#  YAML interface for training multiple model in parallel. Parallel computing parameters can be " \
                   "adjusted at the config tree. \n " \
                   "#  Each model has an own tree and required the definition of a " \
                   "single-model yaml as template. \n" \
                   "#   In the following the keys of the single-yaml can be adjusted.\n" \
                   "#  After adjusting the YAML run the script 'trainFlexNNmulti' to execute the ML task. \n" \
                   "#  For the documentation see: https://proth.pages.rwth-aachen.de/pythonToolBox/ML_utils" \
                   "/getting_started/MultiModelTraining.html \n" \
                   "#  -----------------------------\n"

    return header_single, header_multi
