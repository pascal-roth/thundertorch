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
from stfs_pytoolbox.ML_Utils import metrics


# flexible MLP class
class LightningFlexNN(pl.LightningModule):
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

        self.loss_fn = None
        self.activation_fn = None
        self.min_val_loss = None
        self.final_channel = None

        self.hparams = hparams
        self.check_hparams()
        self.get_default()
        self.get_functions()
        self.set_channels()

        self.explained_variance_train = metrics.ExplainedVariance()
        self.explained_variance_val = metrics.ExplainedVariance()
        self.explained_variance_test = metrics.ExplainedVariance()

        self.layers = []
        self.height = self.hparams.height
        self.width = self.hparams.width
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,)
        self.construct_nn()
        if hasattr(self.hparams, 'MLP_layer'):
            self.layers.append(torch.nn.Flatten())
            self.construct_mlp()
        self.layers = torch.nn.Sequential(*self.layers)

    def construct_nn(self) -> None:
        # Construct all conv layers
        for layer_dict in self.hparams.layers:
            if 'params' in layer_dict:
                self.layers.append(getattr(torch.nn, layer_dict['type'])(**layer_dict['params']))
            else:
                self.layers.append(getattr(torch.nn, layer_dict['type'])())

            if all(hasattr(self.layers[-1], elem) for elem in ['padding', 'stride', 'kernel_size']):
                if isinstance(self.layers[-1].padding, tuple):
                    self.height = int((self.height + 2 * self.layers[-1].padding[0]) /
                                      self.layers[-1].stride[0]) - (self.layers[-1].kernel_size[0] - 1)
                    self.width = int((self.width + 2 * self.layers[-1].padding[1]) /
                                     self.layers[-1].stride[1]) - (self.layers[-1].kernel_size[1] - 1)
                else:
                    self.height = int((self.height + 2 * self.layers[-1].padding) /
                                      self.layers[-1].stride) - (self.height % self.layers[-1].kernel_size)
                    self.width = int((self.width + 2 * self.layers[-1].padding) /
                                     self.layers[-1].stride) - (self.width % self.layers[-1].kernel_size)

            if isinstance(self.layers[-1], self.layer_activation):
                self.layers.append(self.activation_fn)

    def construct_mlp(self) -> None:
        # Construct all MLP layers
        self.layers.append(torch.nn.Linear(self.final_channel * self.height * self.width, self.hparams.MLP_layer['hidden_layer'][0]))
        self.layers.append(self.activation_fn)

        layer_sizes = zip(self.hparams.MLP_layer['hidden_layer'][:-1], self.hparams.MLP_layer['hidden_layer'][1:])

        for h1, h2 in layer_sizes:
            self.layers.append(torch.nn.Linear(h1, h2))
            self.layers.append(self.activation_fn)

        self.layers.append(torch.nn.Linear(self.hparams.MLP_layer['hidden_layer'][-1], self.hparams.MLP_layer['n_out']))

        if hasattr(self.hparams, 'output_activation'):
            self.layers.append(getattr(torch.nn, self.hparams.output_activation)())

    def check_hparams(self) -> None:
        options = self.get_OptionClass()
        OptionClass.checker(input_dict={'hparams': vars(self.hparams)}, option_classes=options)

    def get_default(self) -> None:
        if not hasattr(self.hparams, 'activation'):
            self.hparams.activation = 'ReLU'

        if not hasattr(self.hparams, 'loss'):
            self.hparams.loss = 'MSELoss'

        if not hasattr(self.hparams, 'optimizer'):
            self.hparams.optimizer = {'type': 'Adam', 'params': {'lr': 1e-3}}

        if not hasattr(self.hparams, 'scheduler'):
            self.hparams.scheduler = {'execute': False}

        if not hasattr(self.hparams, 'num_workers'):
            self.hparams.num_workers = 10

        if not hasattr(self.hparams, 'batch'):
            self.hparams.batch = 64

    def get_functions(self) -> None:
        self.activation_fn = getattr(torch.nn, self.hparams.activation)()

        if hasattr(torch.nn, self.hparams.loss):
            self.loss_fn = getattr(torch.nn, self.hparams.loss)()
        else:
            loss_module = getattr(_losses, self.hparams.loss)()
            self.loss_fn = loss_module.forward

    def set_channels(self):
        in_channels = self.hparams.depth
        for i, layer_dict in enumerate(self.hparams.layers):
            if layer_dict['type'] == 'Conv2d' and all(
                    elem not in layer_dict for elem in ['in_channels', 'out_channels']):
                out_channels = layer_dict['params'].pop('channels')
                self.hparams.layers[i]['params']['in_channels'] = in_channels
                self.hparams.layers[i]['params']['out_channels'] = out_channels
                in_channels = out_channels
        self.final_channel = in_channels

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
        x = self.layers(x)

        return x

    def configure_optimizers(self):
        """
        optimizer and lr scheduler

        Returns
        -------
        optimizer       - PyTorch Optimizer function
        scheduler       - PyTorch Scheduler function
        """
        if 'params' in self.hparams.optimizer:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(self.layers.parameters(),
                                                                             **self.hparams.optimizer['params'])
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(self.layers.parameters())

        if self.hparams.scheduler['execute']:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler['type'], 'ReduceLROnPlateau') \
                (optimizer, **self.hparams.scheduler['params'])
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx) -> dict:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long())
        # train_ExpVar = self.explained_variance_train(y_hat, y)
        log = {'train_loss': loss}  #, 'train_ExpVar_step': train_ExpVar}
        results = {'loss': loss, 'log': log}  #, 'train_ExpVar_step': train_ExpVar}
        return results

    # def training_epoch_end(self, outs):
    #     log = {'train_ExpVar_epoch': self.explained_variance_train.compute()}
    #     return {'log': log}

    def validation_step(self, batch, batch_idx) -> dict:
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y.long())
        # self.explained_variance_val(y_hat, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs) -> dict:
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_ExpVar = self.explained_variance_val.compute()
        if self.current_epoch == 0: self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        log = {'avg_val_loss': val_loss}  #, 'val_ExpVar': val_ExpVar}
        pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}  #, 'val_ExpVar': val_ExpVar}
        results = {'log': log, 'val_loss': val_loss, 'progress_bar': pbar}
        return results

    def test_step(self, batch, batch_idx) -> dict:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long())
        # self.explained_variance_test(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs) -> dict:
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # test_ExpVar = self.explained_variance_test.compute()
        log = {'avg_test_loss': test_loss}  #, 'test_ExpVar': test_ExpVar}
        results = {'log': log, 'test_loss': test_loss}  #, 'test_ExpVar': test_ExpVar}
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
        options = {'hparams': OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('depth', dtype=int, required=True)
        options['hparams'].add_key('width', dtype=int, required=True)
        options['hparams'].add_key('height', dtype=int, required=True)
        options['hparams'].add_key('layers', dtype=list, required=True)
        options['hparams'].add_key('MLP_layer', dtype=dict, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=torch.nn)
        options['hparams'].add_key('activation', dtype=str, attr_of=torch.nn)
        options['hparams'].add_key('loss', dtype=str, attr_of=[torch.nn, _losses])
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['layers'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'create_model', 'layers']))
        options['layers'].add_key('type', dtype=str, required=True, attr_of=torch.nn)
        options['layers'].add_key('params', dtype=dict, param_dict=True)

        options['MLP_layer'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'create_model', 'MLP_layer']))
        options['MLP_layer'].add_key('n_out', dtype=int, required=True)
        options['MLP_layer'].add_key('hidden_layer', dtype=list, required=True)

        options['optimizer'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=torch.optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=torch.optim.lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list):
        template = {'Model': {'type': 'LightningFlexNN',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'width': 'int', 'height': 'int', 'depth': 'int',
                                               'layers': [{'type': 'torch.nn module', 'params': {'module_param_1': 'value', 'module_param_2': 'value'}},
                                                          {'type': 'e. g. Conv2d', 'params': {'kernel_size': 3, 'channels': 20}},
                                                          {'type': 'e. g. MaxPool2d', 'params': {'kernel_size': 2}}],
                                               'MLP_layer': {'n_out': 'int', 'hidden_layer': ['int', 'int', '...']},
                                               'output_activation': 'bool (default: False)', 'activation': 'str (default: ReLU)'},
                              'params': {'loss': 'str (default:MSELoss)', 'optimizer': {'type': 'str (default: Adam)',
                                                                                         'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int',
                                                                  'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
