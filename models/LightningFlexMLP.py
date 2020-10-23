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
import pytorch_lightning as pl
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.models import _losses
from stfs_pytoolbox.ML_Utils.utils.utils_option_class import OptionClass


# flexible MLP class
class LightningFlexMLP(pl.LightningModule):
    """
    Create flexMLP as PyTorch LightningModule
    """

    def __init__(self, hparams):
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters (features, labels, lr, activation fn, ...)
        """
        super().__init__()

        self.loss_fn = None
        self.activation_fn = None

        self.hparams = hparams
        self.check_hparams()
        self.get_default()
        self.get_functions()
        self.min_val_loss = None

        # Construct MLP with a variable number of hidden layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.hparams.n_inp, self.hparams.hidden_layer[0])])
        layer_sizes = zip(self.hparams.hidden_layer[:-1], self.hparams.hidden_layer[1:])
        self.layers.extend([torch.nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = torch.nn.Linear(self.hparams.hidden_layer[-1], self.hparams.n_out)

    def check_hparams(self) -> None:
        options = self.get_OptionClass()
        OptionClass.checker(input_dict={'hparams': vars(self.hparams)}, option_classes=options)

    def get_default(self) -> None:
        if not hasattr(self.hparams, 'activation'):
            self.hparams.activation = 'relu'

        if not hasattr(self.hparams, 'loss'):
            self.hparams.loss = 'mse_loss'

        if not hasattr(self.hparams, 'optimizer'):
            self.hparams.optimizer = {'type': 'Adam', 'params': {'lr': 1e-3}}

        if not hasattr(self.hparams, 'scheduler'):
            self.hparams.scheduler = {'execute': False}

        if not hasattr(self.hparams, 'num_workers'):
            self.hparams.num_workers = 10

        if not hasattr(self.hparams, 'batch'):
            self.hparams.batch = 64

        if not hasattr(self.hparams, 'output_relu'):
            self.hparams.output_relu = False

    def get_functions(self):
        self.activation_fn = getattr(torch.nn.functional, self.hparams.activation)

        if hasattr(torch.nn.functional, self.hparams.loss):
            self.loss_fn = getattr(torch.nn.functional, self.hparams.loss)
        else:
            loss_module = getattr(_losses, self.hparams.loss)()
            self.loss_fn = loss_module.forward

    def forward(self, x):
        """
        forward pass through the network

        Parameters
        ----------
        x           - input tensor of the pytorch.nn.Linear layer

        Returns
        -------
        x           - output tensor of the pytorch.nn.Linear layer
        """
        for layer in self.layers:
            x = self.activation_fn(layer(x))

        if self.hparams.output_relu:
            x = torch.nn.functional.relu(self.output(x))
        else:
            x = self.output(x)

        return x

    def configure_optimizers(self):
        """
        optimizer and lr scheduler

        Returns
        -------
        optimizer       - PyTorch Optimizer function
        scheduler       - PyTorch Scheduler function
        """
        params = list(self.layers.parameters()) + list(self.output.parameters())
        if 'params' in self.hparams.optimizer:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(params, **self.hparams.optimizer['params'])
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(params)

        if self.hparams.scheduler['execute']:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler['type'], 'ReduceLROnPlateau')\
                (optimizer, **self.hparams.scheduler['params'])
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        log = {'train_loss': loss}
        results = {'loss': loss, 'log': log}
        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y, y_hat)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.current_epoch == 0: self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        log = {'avg_val_loss': val_loss}
        pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}
        results = {'log': log, 'val_loss': val_loss, 'progress_bar': pbar}
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'avg_test_loss': test_loss}
        results = {'log': log, 'test_loss': test_loss}
        return results

    def hparams_save(self, path) -> None:
        """
        Save hyparams dict to yaml file

        Parameters
        ----------
        path             - path where yaml should be saved
        """
        from pytorch_lightning.core.saving import save_hparams_to_yaml
        save_hparams_to_yaml(path, self.hparams)

    def hparams_update(self, update_dict) -> None:
        """
        Update hyparams dict

        Parameters
        ----------
        update_dict         - dict or namespace object
        """
        from pytorch_lightning.core.saving import update_hparams

        if isinstance(update_dict, Namespace):
            update_dict = vars(update_dict)

        update_hparams(vars(self.hparams), update_dict)
        self.check_hparams()
        self.get_functions()

    # def add_model_specific_args(parent_parser):
    #     parser = argparse.ArgumentParser(parents=[parent_parser])
    #     parser.add_argument('--features', type=list, default=['pode', 'Z', 'H', 'PV'])
    #     parser.add_argument('--labels', type=list, default=['T'])
    #     parser.add_argument('--n_hidden_neurons', nargs='+', type=int, default=[64, 64, 64])
    #     return parser

    @staticmethod
    def get_OptionClass():
        options = {'hparams': OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('n_inp', dtype=int, required=True)
        options['hparams'].add_key('n_out', dtype=int, required=True)
        options['hparams'].add_key('hidden_layer', dtype=list, required=True)
        options['hparams'].add_key('output_relu', dtype=bool)
        options['hparams'].add_key('activation', dtype=str, attr_of=torch.nn.functional)
        options['hparams'].add_key('loss', dtype=str, attr_of=[torch.nn.functional, _losses])
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['optimizer'] = OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=torch.optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexMLP.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=torch.optim.lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list):
        template = {'Model': {'type': 'LightningFlexMLP',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'n_inp': 'int',  'n_out': 'int', 'hidden_layer': '[int, int, int]',
                                               'output_relu': 'bool (default: False)', 'activation':
                                                   'str (default: relu)'},
                              'params': {'loss': 'str (default:mse_loss)', 'optimizer': {'type': 'str (default: Adam)',
                                                                                         'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int', 'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)

