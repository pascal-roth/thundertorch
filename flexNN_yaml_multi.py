#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import logging
# logging.basicConfig(level=logging.DEBUG)
import os
import time
import torch

import torch.multiprocessing as mp
from stfs_pytoolbox.ML_Utils.utils import *


def execute_model(model, argsTrainer, dataLoader):
    train_model(model, dataLoader, argsTrainer)


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


def get_num_processes(argsMulti, argsModels):
    # get nbr of parallel processes for multiprocessing
    nbr_cpu = os.cpu_count()  # nbr of available CPUs
    nbr_gpu = torch.cuda.device_count()  # nbr of available GPUs
    gpu_per_process = 0

    if 'GPU_per_model' in argsMulti:
        assert nbr_gpu != 0, 'GPU per process defined. but NO GPU available!'
        gpu_per_process = argsMulti.pop('GPU_per_model')

        nbr_process = int((nbr_gpu - (nbr_gpu % gpu_per_process)) / gpu_per_process)
        assert nbr_process != 0, f'Not enough GPUs! Model should be trained with {gpu_per_process} GPU(s), but' \
                                 f'available are only {nbr_gpu} GPU(s)'

        logging.info(f'{nbr_process} processes will be executed with {gpu_per_process} GPU(s) per process')

    elif 'CPU_per_model' in argsMulti:
        cpu_per_process = argsMulti.pop('CPU_per_model')

        nbr_process = int((nbr_cpu - (nbr_cpu % cpu_per_process)) / cpu_per_process)
        assert nbr_process != 0, f'Not enough CPUs! Model should be trained with {cpu_per_process} CPU(s), but' \
                                 f'available are only {nbr_cpu} CPU(s)'

        logging.info(f'{nbr_process} processes will be executed with {cpu_per_process} CPU(s) per process')

    else:
        logging.info('Neither "GPU_per_model" nor "CPU_per_model" defined, default setting is selected')
        if nbr_gpu != 0:
            gpu_per_process = 1
            nbr_process = nbr_gpu
            logging.info(f'{nbr_process} processes will be executed with {gpu_per_process} GPU(s) per process')
        else:
            cpu_per_process = 1
            nbr_process = nbr_cpu
            logging.info(f'{nbr_process} processes will be executed with {cpu_per_process} CPU(s) per process')

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


def get_argsModel(argsMulti):
    # filter for models defined in Model_Run list
    if 'Model_run' in argsMulti:
        model_run_list = argsMulti.pop('Model_run')
        assert all(elem in argsMulti for elem in model_run_list), 'Model name included in "Model_run" not found!'
        argsModels = {model_key: argsMulti[model_key] for model_key in model_run_list}
        assert len(argsModels) != 0, 'No models defined in "input_MultiModelTraining.yaml"!'
    else:
        argsModels = argsMulti
        logging.info('No Models excluded! All models selected for training!')

    return argsModels


def main(argsMulti):
    argsModels = get_argsModel(argsMulti)
    nbr_process, list_gpu = get_num_processes(argsMulti, argsModels)
    model_dicts = get_argsDict(argsModels)

    mp_fn = mp.get_context('spawn')
    # lock = mp.Manager().Lock()
    tic1 = time.time()
    processes = []
    ii = 0

    while ii < len(model_dicts):
        models = []
        argsTrainer = []
        dataLoader = []

        for i in range(nbr_process):
            logging.debug('Process started')
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

            ii += 1

        for i in range(nbr_process):
            p = mp_fn.Process(target=execute_model, args=(models[i], argsTrainer[i], dataLoader[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    logging.debug('All models trained')

    tic2 = time.time()
    parallel_forward_pass = tic2 - tic1
    logging.info('Time = {}'.format(parallel_forward_pass))


if __name__ == '__main__':
    args = parse_arguments()
    logger = create_logger(args)
    args_yaml = parse_yaml(args.yaml_path)
    main(args_yaml)
