#######################################################################################################################
# Load arguments of flexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################
# each model will be trained on one GPU, script cannot be used if no gpu is available

# import packages
import yaml
import random
import itertools
import logging
import multiprocessing as mp
import argparse
from stfs_pytoolbox.ML_Utils.flexNN_yaml_single import main


def parse_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_yaml', type=str, default='flexMLP_multi.yaml',
                        help='Name of yaml file to construct Neural Network')
    args = parser.parse_args()

    return yaml.load(open(args.name_yaml), Loader=yaml.FullLoader)


def replace_keys(dictionary, yamlTemplate):
    def recrusion(document, key_list, yamlTemplate):
        if isinstance(document, dict):
            for key, value in document.items():
                key_list.append(key)
                yamlTemplate, key_list = recrusion(document=value, key_list=key_list, yamlTemplate=yamlTemplate)
                key_list = key_list[:-1]
        else:
            try:
                if len(key_list) == 2:
                    yamlTemplate[key_list[0]][key_list[1]] = document
                elif len(key_list) == 3:
                    yamlTemplate[key_list[0]][key_list[1]][key_list[2]] = document
                elif len(key_list) == 4:
                    yamlTemplate[key_list[0]][key_list[1]][key_list[2]][key_list[3]] = document
                else:
                    IndexError('Depth of multi yaml (={}) is out of range of template (=4)'.format(len(key_list)))
            except KeyError:
                raise KeyError('Key {} in multi yaml cannot be found in yamlTemplate'.format(key_list))

        return yamlTemplate, key_list

    yamlTemplate, _ = recrusion(document=dictionary, key_list=list([]), yamlTemplate=yamlTemplate)

    return yamlTemplate


def runModel(args):
    ModelRunName, argsModelRun, list_gpu = args

    yamlTemplate_location = argsModelRun['{}'.format(ModelRunName)].pop('Template')
    yamlTemplate = yaml.load(open(yamlTemplate_location), Loader=yaml.FullLoader)
    yamlTemplate = replace_keys(argsMulti['{}'.format(ModelRunName)], yamlTemplate)

    argsLoader = argparse.Namespace(**yamlTemplate['DataLoader'])
    argsModel = argparse.Namespace(**yamlTemplate['Model'])
    argsTrainer = argparse.Namespace(**yamlTemplate['Trainer'])

    argsTrainer.gpus = list_gpu

    main(argsLoader, argsModel, argsTrainer)


if __name__ == '__main__':
    argsMulti = parse_yaml()

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
        list_gpu = 0

    random.randint(0, 9)

    # filter for models defined in Model_Run list
    model_run_list = argsMulti.pop('Model_run')
    argsModels = {model_key: argsMulti[model_key] for model_key in model_run_list}
    assert len(argsModels) != 0, 'No models defined in "flexMLP_multi.yaml"!'

    pool = mp.Pool(processes=nbr_process)
    pool.map(runModel, zip(argsModels, itertools.repeat(argsModels), list_gpu))
