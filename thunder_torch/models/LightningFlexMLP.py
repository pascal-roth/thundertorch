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
from typing import Optional

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
from thunder_torch.utils.yaml import get_by_path


# flexible MLP class
class LightningFlexMLP(LightningModelBase):
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
        self.construct_mlp(self.hparams.n_inp, self.hparams.hidden_layer, self.hparams.n_out)

        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers_list)

    # def training_step(self, batch, batch_idx) -> dict:
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.loss_fn(y_hat, y)
    #     log = {'train_loss': loss}
    #     results = {'loss': loss, 'log': log}
    #     return results
    #
    # def validation_step(self, batch, batch_idx) -> dict:
    #     x, y = batch
    #     y_hat = self(x)
    #     val_loss = self.loss_fn(y_hat, y)
    #     return {'val_loss': val_loss}
    #
    # def validation_epoch_end(self, outputs) -> dict:
    #     val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     if self.current_epoch == 0: self.min_val_loss = val_loss
    #     if val_loss < self.min_val_loss:
    #         self.min_val_loss = val_loss
    #     log = {'avg_val_loss': val_loss}
    #     pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}
    #     results = {'log': log, 'val_loss': val_loss, 'progress_bar': pbar}
    #     return results
    #
    # def test_step(self, batch, batch_idx) -> dict:
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.loss_fn(y_hat, y)
    #     return {'test_loss': loss}
    #
    # def test_epoch_end(self, outputs) -> dict:
    #     test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     log = {'avg_test_loss': test_loss}
    #     results = {'log': log, 'test_loss': test_loss}
    #     return results

    # # def add_model_specific_args(parent_parser):
    # #     parser = argparse.ArgumentParser(parents=[parent_parser])
    # #     parser.add_argument('--features', type=list, default=['pode', 'Z', 'H', 'PV'])
    # #     parser.add_argument('--labels', type=list, default=['T'])
    # #     parser.add_argument('--n_hidden_neurons', nargs='+', type=int, default=[64, 64, 64])
    # #     return parser

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('n_inp', dtype=int, required=True)
        options['hparams'].add_key('n_out', dtype=int, required=True)
        options['hparams'].add_key('hidden_layer', dtype=list, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('lr', dtype=float)

        options['optimizer'] = OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        """
        Yaml template for LightningFlexMLP
        """
        template = {'Model': {'type': 'LightningFlexMLP',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'n_inp': 'int',  'n_out': 'int', 'hidden_layer': '[int, int, int]',
                                               'output_activation': 'str (default: None)', 'activation':
                                                   'str (default: ReLU)'},
                              'params': {'loss': 'str (default:MSELoss)',
                                         'optimizer': {'type': 'str (default: Adam)',
                                                       'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int',
                                                                  'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        template = get_by_path(template, key_list)

        return yaml.dump(template, sort_keys=False)
