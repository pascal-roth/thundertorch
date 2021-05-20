import torch.nn as nn
import torch
import yaml
from argparse import Namespace
from typing import Optional, List, Callable

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
import thunder_torch as tt


class ResNetDNNBlock(nn.Module):
    def __init__(self, n_neurons: int, activation: Callable[..., torch.Tensor] = torch.nn.ReLU()) -> None:
        super().__init__()

        self.activation = activation
        self.layer1 = nn.Linear(n_neurons, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)

        return out + x


class LightningResMLP(LightningModelBase):
    def __init__(self, hparams: Namespace) -> None:
        """
        Initializes a LightningResMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters
        """
        super(LightningResMLP, self).__init__()

        self.hparams = hparams
        self.check_hparams()
        self.get_default()
        self.get_functions()
        self.min_val_loss: Optional[torch.Tensor] = None

        # add hparams keyword so that model can be easly restored (see utils/general.py::load_model_from_checkpoint)
        self.hparams.model_type = 'LightningResMLP'

        # Construct MLP with a variable number of hidden layers
        self.layers_list = []
        self.layers_list.append(nn.Linear(hparams.n_inp, hparams.hidden_blocks[0]))  # first layer
        # construct hidden residual blocks
        for block in hparams.hidden_blocks:
            self.layers_list.append(ResNetDNNBlock(block, self.activation_fn))
        self.layers_list.append(nn.Linear(self.hparams.hidden_blocks[-1], self.hparams.n_out))   # last layer
        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers_list)

        # define model parameters which should be optimized
        self.optimizer_parameters = self.layers.parameters()

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningResMLP.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('n_inp', dtype=int, required=True)
        options['hparams'].add_key('n_out', dtype=int, required=True)
        options['hparams'].add_key('hidden_blocks', dtype=list, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('lr', dtype=float)

        options['optimizer'] = OptionClass(
            template=LightningResMLP.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(
            template=LightningResMLP.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: List[str]) -> str:
        """
        Yaml template for LightningResMLP
        """
        template = {'Model': {'type': 'LightningFlexMLP',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'n_inp': 'int', 'n_out': 'int', 'hidden_blocks': '[int, int, int]',
                                               'output_activation': 'str (default: None)', 'activation':
                                                   'str (default: ReLU)'},
                              'params': {'loss': 'str (default:MSELoss)',
                                         'optimizer': {'type': 'str (default: Adam)',
                                                       'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int',
                                                                  'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        template = tt.utils.get_by_path(template, key_list)

        return yaml.dump(template, sort_keys=False)
