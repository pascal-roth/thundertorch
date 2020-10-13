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


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def parse_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_yaml', type=str, default='input_MultiModelTraining.yaml',
                        help='Name of yaml file to construct Neural Network')
    args = parser.parse_args()

    return yaml.load(open(args.name_yaml), Loader=yaml.FullLoader)


def replace_keys(dictionary, yamlTemplate):
    def recursion(document, key_list, yamlTemplate):
        if isinstance(document, dict):
            for key, value in document.items():
                key_list.append(key)
                yamlTemplate, key_list = recursion(document=value, key_list=key_list, yamlTemplate=yamlTemplate)
                key_list = key_list[:-1]
        else:
            print(key_list)
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
                raise IndexError('Depth of multi yaml (={}) is out of range of template (=5)'.format(len(key_list)))

        return yamlTemplate, key_list

    yamlTemplate, _ = recursion(document=dictionary, key_list=list([]), yamlTemplate=yamlTemplate)

    return yamlTemplate


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


if __name__ == '__main__':
    argsMulti = parse_yaml()
    main(argsMulti)
