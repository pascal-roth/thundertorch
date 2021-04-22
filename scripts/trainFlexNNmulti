#######################################################################################################################
# Load arguments of multi yaml and execute single yaml
#######################################################################################################################

# import packages
import time
import torch.multiprocessing as mp

from thunder_torch import _logger
from thunder_torch.utils import *


def execute_model(model, argsTrainer: dict, dataLoader) -> None:
    train_model(model, dataLoader, argsTrainer)


def main(argsMulti):
    nbr_processes, list_gpu, model_dicts = config_multi(argsMulti)

    mp_fn = mp.get_context('fork')
    tic1 = time.time()
    processes = []
    ii = 0

    while ii < len(model_dicts):
        models = []
        argsTrainer = []
        dataLoader = []

        for i in range(nbr_processes):
            model_dicts[ii]['trainer']['params']['gpus'] = list_gpu[i]
            model_dicts[ii]['trainer']['params']['process_position'] = i

            check_yaml_version(model_dicts[ii])
            check_yaml_structure(model_dicts[ii])

            if 'config' in model_dicts[ii]:
                check_argsConfig_single(model_dicts[ii]['config'])
                model_dicts[ii]['trainer'] = train_config(model_dicts[ii]['config'], model_dicts[ii]['trainer'])

            check_args(model_dicts[ii])

            models.append(get_model(model_dicts[ii]['model']))
            argsTrainer.append(model_dicts[ii]['trainer'])
            dataLoader.append(get_dataLoader(model_dicts[ii]['dataloader'], models[i]))

            # Increase outer loop counter, need to check if while loop condition is still valid, if not exit inner loop
            ii += 1
            if ii >= len(model_dicts):
                break

        for i in range(nbr_processes):
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
    args_yaml = parse_yaml(args.yaml_path, low_key=False)
    main(args_yaml)
