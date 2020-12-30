#######################################################################################################################
# Initialize usage of Machine Learning Utils per yaml files
#######################################################################################################################

# import packages
import os

from stfs_pytoolbox.ML_Utils import loader
from stfs_pytoolbox.ML_Utils import models
from stfs_pytoolbox.ML_Utils import utils


def initializer():
    # select DataLoader
    msg_Loader = 'Select DataLoader! Implemented Loader are: \n{}\n'.format(loader.__all__)
    name_loader = input(msg_Loader)

    # select model
    msg_model = 'Select Model! Implemented Models are: \n{}\n'.format(models.__all__)
    name_model = input(msg_model)

    # construct single yaml
    loader_dict = getattr(loader, name_loader).yaml_template([])
    model_dict = getattr(models, name_model).yaml_template([])
    trainer_dict = utils.yaml.trainer_yml_template([])
    config_dict = utils.yaml.config_yml_template([])

    with open(f'{os.getcwd()}/{name_loader}_{name_model}.yaml', 'w') as file:
        file.write(f'{config_dict}\n{loader_dict}\n{model_dict}\n{trainer_dict}')

    # construct multi yaml
    multi_dict = utils.yaml.multimodel_training_yml_template([], template=f'{name_loader}_{name_model}.yaml')

    with open(f'{os.getcwd()}/MultiModelTraining.yaml', 'w') as file:
        file.write(f'{multi_dict}')


if __name__ == '__main__':
    initializer()