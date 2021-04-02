#######################################################################################################################
# Load arguments of input_LightningFlexMLP_single.yaml and execute LightningFlexMLP.py
#######################################################################################################################
from stfs_pytoolbox.ML_Utils.utils import *


def main(args_yaml: dict) -> None:
    """
    Load data, load/create LightningModule and train it with the data

    Parameters
    ----------
    args_yaml       - Dict with all input arguments
    """
    check_yaml_version(args_yaml)
    check_yaml_structure(args_yaml)

    if 'config' in args_yaml:
        check_argsConfig_single(args_yaml['config'])
        args_yaml['trainer'] = train_config(args_yaml['config'], args_yaml['trainer'])

    check_args(args_yaml)

    model = get_model(args_yaml['model'])
    dataLoader = get_dataLoader(args_yaml['dataloader'], model=model)
    train_model(model, dataLoader, args_yaml['trainer'])


if __name__ == '__main__':
    args = parse_arguments()
    logger_level(args)
    args_yaml = parse_yaml(args.yaml_path)
    main(args_yaml)
