#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import time
import torch.multiprocessing as mp

from stfs_pytoolbox.ML_Utils import _logger
from stfs_pytoolbox.ML_Utils.utils import *


def execute_model(model, argsTrainer: dict, dataLoader, argsConfig: dict) -> None:
    argsTrainer = train_config(argsConfig, argsTrainer)
    train_model(model, dataLoader, argsTrainer)


def main(argsMulti):
    nbr_processes, list_gpu, model_dicts = config_multi(argsMulti)

    mp_fn = mp.get_context('forkserver')
    tic1 = time.time()
    processes = []
    ii = 0

    while ii < len(model_dicts):
        models = []
        argsTrainer = []
        argsConfig = []
        dataLoader = []

        for i in range(nbr_processes):
            model_dicts[ii]['trainer']['params']['gpus'] = list_gpu[i]
            model_dicts[ii]['trainer']['params']['process_position'] = i

            check_yaml_version(model_dicts[ii])
            check_yaml_structure(model_dicts[ii])

            argsLoader = model_dicts[ii]['dataloader']
            argsModel = model_dicts[ii]['model']
            argsTrainer.append(model_dicts[ii]['trainer'])

            if 'config' in model_dicts[ii]:
                argsConfig.append(model_dicts[ii]['config'])
                check_argsConfig_single(model_dicts[ii]['config'])
            else:
                argsConfig.append([])

            check_args(argsModel, argsLoader, argsTrainer[i])

            models.append(get_model(argsModel))
            dataLoader.append(get_dataLoader(argsLoader, models[i]))

            # Increase outer loop counter, need to check if while loop condition is still valid, if not exit inner loop
            ii += 1
            if ii >= len(model_dicts):
                break

        for i in range(nbr_processes):
            p = mp_fn.Process(target=execute_model, args=(models[i], argsTrainer[i], dataLoader[i], argsConfig[i]))
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
