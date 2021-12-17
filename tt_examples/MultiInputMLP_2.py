#######################################################################################################################
# flexible MLP model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import Optional, Tuple, Union, List

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
import thunder_torch as tt


# flexible MLP class
class LightningFlexMLPTwoLevelInput(LightningModelBase):
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

        # Construct MLP with a variable number of hidden layers
        self.layers_list = []
        self.construct_mlp(self.hparams.n_inp_1, self.hparams.hidden_layer_1, self.hparams.n_out_1)
        self.layers_list.append(getattr(torch.nn, self.hparams.activation)())
        self.layers_1 = torch.nn.Sequential(*self.layers_list)

        self.layers_list = []
        self.construct_mlp(self.hparams.n_inp_2, self.hparams.hidden_layer_2, self.hparams.n_out_2)
        self.layers_2 = torch.nn.Sequential(*self.layers_list)

    def get_optimizer_parameters(self) -> Union[torch.Generator, List[torch.Generator]]:
        params = []
        params += list(self.layers_1.parameters())
        params += list(self.layers_2.parameters())
        return params

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        x = self.layers_1(x_1)
        x = torch.cat((x, x_2), dim=1)
        x = self.layers_2(x)
        return torch.squeeze(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Training step of the network
        """
        x_1, x_2, y = batch
        y_hat = self(x_1, x_2)
        loss = self.loss_fn(y_hat, y)
        log = {'train_loss': loss}
        results = {'loss': loss, 'log': log}
        return results

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Validation step of the network
        """
        x_1, x_2, y = batch
        y_hat = self(x_1, x_2)
        loss = self.loss_fn(y_hat, y)
        return {'val_loss': loss}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        """
        x_1, x_2, y = batch
        y_hat = self(x_1, x_2)
        loss = self.loss_fn(y_hat, y)
        return {'test_loss': loss}

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexMLPTwoLevelInput.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('n_inp_1', dtype=int, required=True)
        options['hparams'].add_key('n_inp_2', dtype=int, required=True)
        options['hparams'].add_key('n_out_1', dtype=int, required=True)
        options['hparams'].add_key('n_out_2', dtype=int, required=True)
        options['hparams'].add_key('hidden_layer_1', dtype=list, required=True)
        options['hparams'].add_key('hidden_layer_2', dtype=list, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('lr', dtype=float)

        options['optimizer'] = OptionClass(template=LightningFlexMLPTwoLevelInput.yaml_template(['Model', 'params',
                                                                                                 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexMLPTwoLevelInput.yaml_template(['Model', 'params',
                                                                                                 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template for LightningFlexMLP
        """
        template = {'Model': {'type': 'LightningFlexMLPTwoLevelInput',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'n_inp_1': 'int', 'n_out_1': 'int', 'hidden_layer_1': '[int, int, int]',
                                               'n_inp_2': 'int', 'n_out_2': 'int', 'hidden_layer_2': '[int, int, int]',
                                               'output_activation': 'str (default: None)',
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
