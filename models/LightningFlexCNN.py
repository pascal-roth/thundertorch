#######################################################################################################################
# flexible CNN model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################

# import packages
import torch
import yaml
import pytorch_lightning as pl
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.models import _losses
from stfs_pytoolbox.ML_Utils.utils.utils_option_class import OptionClass


# flexible MLP class
class LightningFlexCNN(pl.LightningModule):
    """
    Create flexMLP as PyTorch LightningModule

    Convolutional Layer Parameters:
        - in_channels (int) – Number of channels in the input image
        - out_channels (int) – Number of channels produced by the convolution
        - kernel_size (int or tuple) – Size of the convolving kernel
        - stride (int or tuple, optional) – Stride of the convolution. Default: 1
        - padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        - padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        - dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        - groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        - bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

    Normalization Layer Parameters:
        - normalized_shape (int or list or torch.Size) – input shape from an expected input of size
        - eps – a value added to the denominator for numerical stability. Default: 1e-5
        - elementwise_affine – a boolean value that when set to True, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.

    MaxPool Layer Parameters:
        - kernel_size – the size of the window to take a max over
        - stride – the stride of the window. Default value is kernel_size
        - padding – implicit zero padding to be added on both sides
        - dilation – a parameter that controls the stride of elements in the window
        - return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
        - ceil_mode – when True, will use ceil instead of floor to compute the output shape
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
        self.get_default()
        self.set_channels()
        self.min_val_loss = None

        # Construct all conv layers
        self.conv_layers = torch.nn.ModuleList()

        for layer_dict in self.hparams.conv_layer:
            if 'params' in layer_dict:
                self.conv_layers.extend([getattr(torch.nn, layer_dict['type'])(**layer_dict['params'])])
            else:
                self.conv_layers.extend([getattr(torch.nn, layer_dict['type'])])

        # Construct all MLP layers
        self.mlp_layers = torch.nn.ModuleList([torch.nn.Linear('last channel * x_dim * y_dim', self.hparams.hidden_layer[0])])
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

    def set_channels(self):
        in_channels = self.hparams.n_inp  # TODO: eigentlich pop aber wie wenn dann der checkpoint geladen wird
        for i, layer_dict in enumerate(self.hparams.conv_layer):
            if layer_dict['type'] == 'Conv2d':
                out_channels = layer_dict['params'].pop('channels')
                self.hparams.conv_layer[i]['params']['in_channels'] = in_channels
                self.hparams.conv_layer[i]['params']['out_channels'] = out_channels
                in_channels = out_channels

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
        if hasattr(torch.nn.functional, self.hparams.loss):
            loss = getattr(torch.nn.functional, self.hparams.loss)(y_hat, y)
        else:
            loss = getattr(_losses, self.hparams.loss).loss_fn(y_hat, y)
        return loss

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
            x = activation_fn(layer(x))  # TODO: add that only if Con2d and Linear layer activation should be performed

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

    # def add_model_specific_args(parent_parser):
    #     parser = argparse.ArgumentParser(parents=[parent_parser])
    #     parser.add_argument('--features', type=list, default=['pode', 'Z', 'H', 'PV'])
    #     parser.add_argument('--labels', type=list, default=['T'])
    #     parser.add_argument('--n_hidden_neurons', nargs='+', type=int, default=[64, 64, 64])
    #     return parser

    @staticmethod
    def get_OptionClass():
        options = {'hparams': OptionClass()}  # TODO: add template, when its finished
        options['hparams'].add_key('n_inp', dtype=int, required=True)
        options['hparams'].add_key('conv_layer', dtype=list, required=True)
        options['hparams'].add_key('MLP_layer', dtype=dict, required=True)
        options['hparams'].add_key('output_relu', dtype=bool)
        options['hparams'].add_key('activation', dtype=str, attr_of=torch.nn.functional)
        options['hparams'].add_key('loss', dtype=str, attr_of=[torch.nn.functional, _losses])
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['conv_layer'] = OptionClass()
        options['conv_layer'].add_key('type', dtype=str, required=True, attr_of=torch.nn)
        options['conv_layer'].add_key('params', dtype=dict, param_dict=True)  # TODO: check if NormLayer needs params -> otherwise not required

        options['MLP_layer'] = OptionClass()
        options['MLP_layer'].add_key('n_out', dtype=int, required=True)
        options['MLP_layer'].add_key('hidden_layer', dtype=list, required=True)

        options['optimizer'] = OptionClass()
        options['optimizer'].add_key('type', dtype=str, attr_of=torch.optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass()
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

