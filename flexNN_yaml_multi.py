#######################################################################################################################
# Load arguments of flexMLP_single.yaml and execute flexMLP_pl.py
#######################################################################################################################
# each model will be trained on one GPU, script cannot be used if no gpu is available

# import packages
import yaml
import itertools
import numpy as np
import multiprocessing as mp
from argparse import Namespace
from flexNN_yaml_single import main


def parse_yaml(name):
    flexMLP_yaml = open(name)
    return yaml.load(flexMLP_yaml, Loader=yaml.FullLoader)


def get_all_keys(dictionary):

    def recrusion(document, key_list, n_row, n_column):
        if isinstance(document, dict):
            for key, value in document.items():
                key_list[n_row, n_column] = key
                n_column += 1
                key_list, n_row, n_column = recrusion(document=value, key_list=key_list, n_row=n_row, n_column=n_column)
                n_column -= 1
        else:
            key_list = np.append(key_list, np.zeros([1, 3], dtype='<U15'), axis=0)
            n_row += 1
            key_list[n_row, :(n_column-1)] = key_list[n_row - 1, :(n_column-1)]
        return key_list, n_row, n_column

    key_list, _, _ = recrusion(document=dictionary, key_list=np.zeros([1, 3], dtype='<U15'), n_row=0, n_column=0)
    key_list = key_list[:(len(key_list)-1), :]
    return key_list


def runModel(args):

    ModelRunName, argsMulti, list_gpu = args
    argsSingle = parse_yaml(name='flexMLP_single.yaml')

    keys = get_all_keys(argsMulti['{}'.format(ModelRunName)])

    for i in range(len(keys)):
        if keys[i, 2] == '':
            argsSingle[keys[i, 0]][keys[i, 1]] = argsMulti['{}'.format(ModelRunName)][keys[i, 0]][keys[i, 1]]
        else:
            argsSingle[keys[i, 0]][keys[i, 1]][keys[i, 2]] = argsMulti['{}'.format(ModelRunName)][keys[i, 0]][keys[i, 1]][keys[i, 2]]

    argsLoader = Namespace(**argsSingle['TabularLoader'])
    argsModel = Namespace(**argsSingle['flexMLP_pl'])
    argsTrainer = Namespace(**argsSingle['pl.Trainer'])

    argsTrainer.gpus = [list_gpu]
    main(argsLoader, argsModel, argsTrainer)


if __name__ == '__main__':
    argsMulti = parse_yaml(name='flexMLP_multi.yaml')

    if len(argsMulti) < 4 and len(argsMulti) != 0:
        n_gpu = len(argsMulti)
    elif len(argsMulti) >= 4:
        n_gpu = 4
    else:
        raise KeyError('No models defined in "flexMLP_multi.yaml"!')

    ar_gpu = [0, 1, 2, 3]

    pool = mp.Pool(processes=n_gpu)
    pool.map(runModel, zip(argsMulti, itertools.repeat(argsMulti), ar_gpu))
