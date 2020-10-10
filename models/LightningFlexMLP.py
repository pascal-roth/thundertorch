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
from stfs_pytoolbox.ML_Utils.losses import RelativeMSELoss


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

        self.hparams = hparams
        self.check_hparams()
        self.min_val_loss = None

        # Construct MLP with a variable number of hidden layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.hparams.n_inp, self.hparams.hidden_layer[0])])
        layer_sizes = zip(self.hparams.hidden_layer[:-1], self.hparams.hidden_layer[1:])
        self.layers.extend([torch.nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = torch.nn.Linear(self.hparams.hidden_layer[-1], self.hparams.n_out)

    def check_hparams(self) -> None:
        # check model building hparams -> do not have default values
        assert hasattr(self.hparams, 'n_inp'), 'Definition of input dimension is missing! Define "n_inp" as type(int)'
        assert isinstance(self.hparams.n_inp, int), 'Size of input layer has to be of type int'
        assert hasattr(self.hparams, 'n_out'), 'Definition of output dimension is missing! Define "n_out" as type(int)'
        assert isinstance(self.hparams.n_out, int), 'Size of output layer has to be of type int'
        assert hasattr(self.hparams, 'hidden_layer'), 'Definition of hidden layer dimension(s) is missing! ' \
                                                      'Define "hidden_layer" as list of int(s)'
        if not isinstance(self.hparams.hidden_layer, list): self.hparams.hidden_layer = [self.hparams.hidden_layer]
        assert all(isinstance(elem, int) for elem in self.hparams.hidden_layer), 'Size of hidden layer must be type int'

        # check functions
        if hasattr(self.hparams, 'activation'):
            assert isinstance(self.hparams.activation, str), 'Activation function type has to be of type str'
            assert hasattr(torch.nn.functional, self.hparams.activation), ('Activation function {} not implemented in '
                                                                           'torch'.format(self.hparams.activation))
        else:
            self.hparams.activation = 'relu'

        if hasattr(self.hparams, 'loss'):
            assert isinstance(self.hparams.loss, str), 'Loss function type has to be of type str'
            assert hasattr(torch.nn.functional, self.hparams.loss), 'Loss function {} not implemented in ' \
                                                                    'torch'.format(self.hparams.loss)
        else:
            self.hparams.loss = 'mse_loss'

        if hasattr(self.hparams, 'optimizer'):
            assert self.hparams.optimizer, 'Optimizer params are missing. Attach dict with structure: \n{}'.\
                format(self.yaml_template(['params', 'optimizer']))
            assert isinstance(self.hparams.optimizer['type'], str), 'Optimizer function type has to be of type str'
            assert hasattr(torch.optim, self.hparams.optimizer['type']), 'Optimizer function {} not implemented in ' \
                                                                         'torch'.format(self.hparams.optimizer['type'])
        else:
            self.hparams.optimizer = {'type': 'Adam', 'params': {'lr': 1e-3}}

        if hasattr(self.hparams, 'scheduler'):
            assert self.hparams.scheduler, 'Scheduler params are missing. Attach dict with structure: \n{}'.\
                format(self.yaml_template(['params', 'scheduler']))
            if self.hparams.scheduler['execute']:
                assert isinstance(self.hparams.scheduler['type'], str), 'Scheduler function type has to be of type str'
                assert hasattr(torch.optim.lr_scheduler, self.hparams.scheduler['type']), \
                    'Scheduler function {} not implemented in torch'.format(self.hparams.scheduler['type'])
        else:
            self.hparams.scheduler = {'execute': False}

        # introduce default values
        if not hasattr(self.hparams, 'num_workers'):
            self.hparams.num_workers = 10
        else:
            assert isinstance(self.hparams.num_workers, int), 'Num_workers has to be of type int, not {}!'. \
                format(type(self.hparams.num_workers))

        if not hasattr(self.hparams, 'batch'):
            self.hparams.batch = 64
        else:
            assert isinstance(self.hparams.batch, int), 'Batch size has to be of type int, not {}!'. \
                format(type(self.hparams.batch))

        if not hasattr(self.hparams, 'output_relu'):
            self.hparams.output_relu = False
        else:
            assert isinstance(self.hparams.output_relu, bool), 'Output_relu has to be of type bool, not {}!'. \
                format(type(self.hparams.output_relu))

    def loss_fn(self, y, y_hat):
        """
        compute loss

        Parameters
        ----------
        y           - target tensor of network
        y_hat       - tensor output of network

        Returns
        -------
        loss        - float
        """
        loss_fn = getattr(torch.nn.functional, self.hparams.loss, 'mse_loss')
        return loss_fn(y_hat, y)

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
            activation_fn = getattr(torch.nn.functional, self.hparams.activation)
            x = activation_fn(layer(x))

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

    # def add_model_specific_args(parent_parser):
    #     parser = argparse.ArgumentParser(parents=[parent_parser])
    #     parser.add_argument('--features', type=list, default=['pode', 'Z', 'H', 'PV'])
    #     parser.add_argument('--labels', type=list, default=['T'])
    #     parser.add_argument('--n_hidden_neurons', nargs='+', type=int, default=[64, 64, 64])
    #     return parser

    @staticmethod
    def yaml_template(key_list):
        template = {'type': 'LightningFlexMLP',
                    'source': 'load/ create',
                    'load_model': {'path': 'name.ckpt'},
                    'create_model': {'n_inp': int,  'n_out': int, 'hidden_layer': [int, int, int],
                                     'output_relu': 'bool (default: False)', 'activation': 'relu'},
                    'params': {'loss': 'mse_loss', 'optimizer': {'type': 'Adam', 'params': {'lr': 0.001}},
                               'scheduler': {'execute': ' bool (default: False)', 'type': 'name (ReduceLROnPlateau)',
                                             'params': {'cooldown': int, 'patience': int, 'min_lr': float}},
                               'num_workers': int, 'batch': int}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        print(yaml.dump(template, sort_keys=False))

