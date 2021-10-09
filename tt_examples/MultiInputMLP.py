#######################################################################################################################
# flexible MLP model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################
# Problem with Tensorboard (officially available only with pytorch 1.3
#    - add function hparams to torch/utils/tensorboard/summary.py
#    - remove version check in pytorch_lightning/logggers/tensorboard.py
# idea define own logger where all code is copied and just those changes implemented
# access saved data by script and execute plotting: https://www.tensorflow.org/tensorboard/dataframe_api

# import packages
import torch
import yaml
from argparse import Namespace
from typing import Optional, Tuple, Union, List

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.models._utils import Cat
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
import thunder_torch as tt


# flexible MLP class
class LightningFlexMLPMultiInput(LightningModelBase):
    """
    Create flexMLP as PyTorch LightningModule

    Hyperparameters of the model
    ----------------------------
    - n_inp:                int         Input dimension (required)
    - n_out:                int         Output dimension (required)
    - hidden_layer:         list        List of hidden layers with number of hidden neurons as layer entry (required)
    - activation:           str         activation fkt that is included in torch.nn (default: ReLU)
    - loss:                 str         loss fkt that is included in torch.nn (default: MSELoss)
    - optimizer:            dict        dict including optimizer fkt type and possible parameters, optimizer has to be
                                        included in torch.optim (default: {'type': Adam, 'params': {'lr': 1e-3}})
    - scheduler:            dict        dict including execute flag, scheduler fkt type and possible parameters,
                                        scheduler
                                        has to be included in torch.optim.lr_scheduler (default: {'execute': False})
    - num_workers:          int         number of workers in DataLoaders (default: 10)
    - batch:                int         batch size of DataLoaders (default: 64)
    - output_activation:    str         activation fkt  (default: False)
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters
        """
        super().__init__()

        self.hparams = hparams
        self.check_hparams()
        self.get_default()
        self.get_functions()
        self.min_val_loss: Optional[torch.Tensor] = None

        # Construct MLP for the single input
        self.layers_list = []
        self.construct_mlp(self.hparams.mlp_single['n_inp'], self.hparams.mlp_single['hidden_layer'],
                           self.hparams.mlp_single['n_out'])
        self.layers_list.append(self.activation_fn)

        self.mlp_single = torch.nn.ModuleList([torch.nn.Sequential(*self.layers_list) for _ in
                                               range(self.hparams.multi_input)])

        # construct MLP for combined input
        self.layers_list = []
        self.layers_list.append(Cat(dim=1))
        self.construct_mlp(self.hparams.mlp_combined['n_inp'], self.hparams.mlp_combined['hidden_layer'],
                           self.hparams.mlp_combined['n_out'])

        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers_list)

    def get_optimizer_parameters(self) -> Union[torch.Generator, List[torch.Generator]]:
        params = []
        if hasattr(self, 'mlp_single'):
            for i in range(len(self.mlp_single)):
                params += list(self.mlp_single[i].parameters())
        params += list(self.layers.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_list = [self.mlp_single[i](x[:, i, :]) for i in range(x.shape[1])]
        x = self.layers(input_list)
        x = torch.squeeze(x)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Training step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        log = {'train_loss': loss}
        hiddens = {'inputs': x[:, 1, :].detach(), 'preds': y_hat.detach(),
                   'targets': y.detach()}  # TODO general solution
        results = {'loss': loss, 'log': log, 'hiddens': hiddens}
        return results

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Validation step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        hiddens = {'inputs': x[:, 1, :].detach(), 'preds': y_hat.detach(),
                   'targets': y.detach()}  # TODO general solution
        return {'val_loss': loss, 'hiddens': hiddens}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        hiddens = {'inputs': x[:, 1, :].detach(), 'preds': y_hat.detach(),
                   'targets': y.detach()}  # TODO general solution
        return {'test_loss': loss, 'hiddens': hiddens}

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexMLPMultiInput.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('mlp_single', dtype=dict, required=True)
        options['hparams'].add_key('mlp_combined', dtype=dict, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('multi_input', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('lr', dtype=float)

        options['mlp_single'] = OptionClass(template=LightningFlexMLPMultiInput.yaml_template(['Model', 'create_model', 'mlp_single']))
        options['mlp_single'].add_key('n_inp', dtype=int, required=True)
        options['mlp_single'].add_key('n_out', dtype=int, required=True)
        options['mlp_single'].add_key('hidden_layer', dtype=list, required=True)

        options['mlp_combined'] = options['mlp_single']

        options['optimizer'] = OptionClass(template=LightningFlexMLPMultiInput.yaml_template(['Model', 'create_model', 'mlp_combined']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexMLPMultiInput.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template for LightningFlexMLPMultiInput
        """
        template = {'Model': {'type': 'LightningFlexMLPMultiInput',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'mlp_single': {'n_inp': 'int',  'n_out': 'int',
                                                              'hidden_layer': '[int, int, int]'},
                                               'mlp_combined': {'n_inp': 'int', 'n_out': 'int',
                                                                'hidden_layer': '[int, int, int]'},
                                               'output_activation': 'str (default: None)',
                                               'multi_input': 'int',
                                               'activation': 'str (default: ReLU)'},
                              'params': {'loss': 'str (default:MSELoss)',
                                         'optimizer': {'type': 'str (default: Adam)',
                                                       'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int',
                                                                  'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        template = tt.utils.get_by_path(template, key_list)

        return yaml.dump(template, sort_keys=False)
