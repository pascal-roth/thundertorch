#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import os
import time
import torch
import torch.multiprocessing as mp

from stfs_pytoolbox.ML_Utils import _logger
from stfs_pytoolbox.ML_Utils.utils import *
from stfs_pytoolbox.ML_Utils.utils.yaml import multimodel_training_yml_template


def execute_model(model, argsTrainer, dataLoader):
    train_model(model, dataLoader, argsTrainer)


def config(argsMulti):
    argsModels, argsConfig = get_argsModel(argsMulti)
    nbr_process, list_gpu = get_num_processes(argsConfig, argsModels)
    model_dicts = get_argsDict(argsModels)
    return nbr_process, list_gpu, model_dicts


def get_argsModel(argsMulti):
    # check if config dict in yaml
    if 'config' in argsMulti:
        argsConfig = argsMulti.pop('config')
        check_config(argsConfig)
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


def check_config(argsConfig: dict) -> None:
    options = {'config': OptionClass(template=multimodel_training_yml_template(['config']))}
    options['config'].add_key('nbr_process', dtype=int)
    options['config'].add_key('GPU_per_model', dtype=int, mutually_exclusive=['CPU_per_model'])
    options['config'].add_key('CPU_per_model', dtype=int, mutually_exclusive=['GPU_per_model'])
    options['config'].add_key('model_run', dtype=list)

    OptionClass.checker(input_dict={'config': argsConfig}, option_classes=options)


def get_argsDict(argsModels):
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


def get_num_processes(argsConfig, argsModels):
    # get nbr of parallel processes for multiprocessing
    nbr_cpu = os.cpu_count()  # nbr of available CPUs
    nbr_gpu = torch.cuda.device_count()  # nbr of available GPUs
    gpu_per_process = 0

    if 'GPU_per_model' in argsConfig:
        assert nbr_gpu != 0, 'GPU per process defined. but NO GPU available!'
        gpu_per_process = argsConfig.pop('GPU_per_model')

        nbr_process = int((nbr_gpu - (nbr_gpu % gpu_per_process)) / gpu_per_process)
        assert nbr_process != 0, f'Not enough GPUs! Model should be trained with {gpu_per_process} GPU(s), but' \
                                 f'available are only {nbr_gpu} GPU(s)'

        _logger.debug(f'{nbr_process} processes can be executed with {gpu_per_process} GPU(s) per process')

    elif 'CPU_per_model' in argsConfig:
        cpu_per_process = argsConfig.pop('CPU_per_model')

        nbr_process = int((nbr_cpu - (nbr_cpu % cpu_per_process)) / cpu_per_process)
        assert nbr_process != 0, f'Not enough CPUs! Model should be trained with {cpu_per_process} CPU(s), but' \
                                 f'available are only {nbr_cpu} CPU(s)'

        _logger.debug(f'{nbr_process} processes can be executed with {cpu_per_process} CPU(s) per process')

    else:
        _logger.debug('Neither "GPU_per_model" nor "CPU_per_model" defined, default setting is selected')
        if nbr_gpu != 0:
            gpu_per_process = 1
            nbr_process = nbr_gpu
            _logger.debug(f'{nbr_process} processes can be executed with {gpu_per_process} GPU(s) per process')
        else:
            cpu_per_process = 1
            nbr_process = nbr_cpu
            _logger.debug(f'{nbr_process} processes can be executed with {cpu_per_process} CPU(s) per process')

    if 'nbr_process' in argsConfig:
        _logger.debug(f'Config nbr_process {argsConfig["nbr_process"]} is compared with maximal possible number of '
                      f'processes {nbr_process}, minimum is selected')
        nbr_process = min(nbr_process, argsConfig['nbr_process'])

    nbr_process = min(len(argsModels), nbr_process)

    if gpu_per_process != 0 and nbr_gpu != 0:
        list_gpu = []
        gpu_available = list(range(0, nbr_gpu))
        for i in range(nbr_process):
            list_gpu.append(gpu_available[0:gpu_per_process])
            del gpu_available[0:gpu_per_process]
    else:
        list_gpu = [0 for x in range(nbr_process)]

    return nbr_process, list_gpu


def main(argsMulti):
    nbr_process, list_gpu, model_dicts = config(argsMulti)

    mp_fn = mp.get_context('forkserver')
    tic1 = time.time()
    processes = []
    ii = 0

    while ii < len(model_dicts):
        models = []
        argsTrainer = []
        dataLoader = []

        for i in range(nbr_process):
            model_dicts[ii]['Trainer']['params']['gpus'] = list_gpu[i]
            model_dicts[ii]['Trainer']['params']['process_position'] = i

            check_yaml_version(model_dicts[ii])
            check_yaml_structure(model_dicts[ii])

            argsLoader = model_dicts[ii]['DataLoader']
            argsModel = model_dicts[ii]['Model']
            argsTrainer.append(model_dicts[ii]['Trainer'])

            check_args(argsModel, argsLoader, argsTrainer[i])

            models.append(get_model(argsModel))
            dataLoader.append(get_dataLoader(argsLoader, models[i]))

            # Increase outer loop counter, need to check if while loop condition is still valid, if not exit inner loop
            ii += 1
            if ii >= len(model_dicts):
                break

        for i in range(nbr_process):
            p = mp_fn.Process(target=execute_model, args=(models[i], argsTrainer[i], dataLoader[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    _logger.info('All models trained')

    tic2 = time.time()
    parallel_forward_pass = tic2 - tic1
    _logger.info('Time = {}'.format(parallel_forward_pass))


if __name__ == '__main__':
    args = parse_arguments()
    logger_level(args)
    args_yaml = parse_yaml(args.yaml_path)
    main(args_yaml)
