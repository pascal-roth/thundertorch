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


def execute_model(model_dict, lock):
    check_yaml_version(model_dict)
    check_yaml_structure(model_dict)

    argsLoader = model_dict['DataLoader']
    argsModel = model_dict['Model']
    argsTrainer = model_dict['Trainer']

    check_args(argsModel, argsLoader, argsTrainer)

    model = get_model(argsModel)

    with lock:
        logging.debug('Lock DataLoader active')
        dataLoader = get_dataLoader(argsLoader, model)

    logging.debug('Lock DataLoader deactivated')

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
    if 'Nbr_processes' in argsMulti:
        nbr_process = argsMulti.pop('Nbr_processes')
        assert nbr_process != 0, 'Number of processes must be > 0'

        if nbr_gpu != 0:
            assert nbr_process <= nbr_gpu, 'The number of intended processes exceeds the number of available GPUs!'
        else:
            logging.info('No GPU available!')
            assert nbr_process <= nbr_cpu, 'The number of intended processes exceeds the number of available CPUs!'

    else:
        if nbr_gpu != 0:
            nbr_process = nbr_gpu
        else:
            nbr_process = nbr_cpu

    gpu_per_process = argsMulti.pop('GPU_per_model', 1)
    nbr_process = min(nbr_process, len(list(argsModels)))

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
    lock = mp.Manager().Lock()
    tic1 = time.time()
    processes = []
    ii = 0

    while ii < len(model_dicts):
        for i in range(nbr_process):
            logging.debug('Process started')
            model_dicts[ii]['Trainer']['params']['gpus'] = list_gpu[i]
            model_dicts[ii]['Trainer']['params']['process_position'] = i
            p = mp_fn.Process(target=execute_model, args=(model_dicts[ii], lock))
            processes.append(p)
            p.start()
            ii += 1
        for p in processes:
            p.join()

        # wait for all processes to finish --------------------------------
        while processes:
            for i, process_data in enumerate(processes):
                if not process_data.is_alive():
                    logging.debug(f'process finished')
                    # remove from processes
                    p = processes.pop(i)
                    del p
                    # start new search
                    break
        logging.debug('Round of processes finished')

    logging.debug('All models trained')

    tic2 = time.time()
    parallel_forward_pass = tic2 - tic1
    logging.info('Time = {}'.format(parallel_forward_pass))


if __name__ == '__main__':
    argsMulti = parse_yaml()
    main(argsMulti)
