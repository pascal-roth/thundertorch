#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################
from stfs_pytoolbox.ML_Utils.utils import *


def main(args_yaml):
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    args_yaml       - Dict with all input arguments
    """
    check_yaml_structure(args_yaml)

    argsLoader = args_yaml['DataLoader']
    argsModel = args_yaml['Model']
    argsTrainer = args_yaml['Trainer']

    check_args(argsModel, argsLoader, argsTrainer)

    model = get_model(argsModel)
    dataLoader = get_dataLoader(argsLoader, model)
    train_model(model, dataLoader, argsTrainer)


if __name__ == '__main__':
    args_yaml = parse_yaml()
    main(args_yaml)
