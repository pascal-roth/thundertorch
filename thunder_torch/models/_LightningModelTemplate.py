#######################################################################################################################
# Template to construct Neural Network Models using PyTorch Lightning
#######################################################################################################################

# import packages
import yaml
from argparse import Namespace

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim


class LightningModelTemplate(LightningModelBase):
    """
    Create Model as PyTorch LightningModule
    """

    def __init__(self, hparams):
        """
        Initializes a model with hyperparameters (all parameters included in hparams Namespace are saved automatically
        in the model checkpoint!)
        """
        super().__init__()

        self.hparams = hparams
        self.check_hparams()
        self.get_defaults()
        self.min_val_loss = None

        # code to construct the model
        # store layers in self.layers or change this flag in optimization function
        # store output layer in self.output or change this flag in optimization

    @staticmethod
    def get_OptionClass():
        options = {'hparams': OptionClass(template=LightningModelTemplate.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('model_param_1', required=True)
        options['hparams'].add_key('model_param_2', required=True)
        options['hparams'].add_key('model_param_3', required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('lr', dtype=float)

        options['optimizer'] = OptionClass(template=LightningModelTemplate.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningModelTemplate.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

    @staticmethod
    def yaml_template(key_list):
        template = {'Model': {'type': 'LightningModelTemplate',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'some_construction_parameters': 'corresponding datatypes',
                                               'output_relu': 'bool (default: False)', 'activation': 'relu'},
                              'params': {'loss': 'mse_loss', 'optimizer': {'type': 'Adam', 'params': {'lr': 1.e-3}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'param_1': 'int', 'param_2': 'int'}},
                                         'num_workers': 'int', 'batch': 'int'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
