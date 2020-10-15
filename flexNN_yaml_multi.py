#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################

# import packages
import yaml
import itertools
import argparse
import logging
import multiprocessing.pool
from stfs_pytoolbox.ML_Utils.flexNN_yaml_single import main as execute_run
from stfs_pytoolbox.ML_Utils.utils import *


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def runModel(args):
    ModelRunName, argsModelRun, list_gpu = args

    assert 'Template' in argsModelRun[ModelRunName], 'Definition of a Template necessary to change model keys!'
    yamlTemplate_location = argsModelRun[ModelRunName].pop('Template')
    yamlTemplate = yaml.load(open(yamlTemplate_location), Loader=yaml.FullLoader)
    yamlModelRun = replace_keys(argsModelRun[ModelRunName], yamlTemplate)

    yamlModelRun['Trainer']['params']['gpus'] = list_gpu

    execute_run(yamlModelRun)


def main(argsMulti):
    # filter for models defined in Model_Run list
    if 'Model_run' in argsMulti:
        model_run_list = argsMulti.pop('Model_run')
        assert all(elem in argsMulti for elem in model_run_list), 'Model name included in "Model_run" not found!'
        argsModels = {model_key: argsMulti[model_key] for model_key in model_run_list}
        assert len(argsModels) != 0, 'No models defined in "input_MultiModelTraining.yaml"!'
    else:
        argsModels = argsMulti
        logging.info('No Models excluded! All models selected for training!')

    # get nbr of parallel processes for multiprocessing
    nbr_process = argsMulti.pop('Nbr_processes', 4)
    # nbr_process = os.cpu_count()  # nbr of available CPUs
    # nbr_process = torch.cuda.device_count()  # nbr of available GPUs
    gpu_per_process = argsMulti.pop('GPU_per_model', 1)
    assert nbr_process != 0, 'Number of processes must be > 0'
    if gpu_per_process != 0:
        list_gpu = []
        gpu_available = list(range(0, nbr_process))
        for i in range(int(nbr_process / gpu_per_process)):
            list_gpu.append(gpu_available[0:gpu_per_process])
            del gpu_available[0:gpu_per_process]
    else:
        list_gpu = itertools.repeat(0)

    pool = MyPool(nbr_process)
    pool.map(runModel, zip(argsModels, itertools.repeat(argsModels), list_gpu))
    pool.close()
    pool.join()


if __name__ == '__main__':
    argsMulti = parse_yaml()
    main(argsMulti)
