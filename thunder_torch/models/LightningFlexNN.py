#######################################################################################################################
# flexible CNN model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import Optional

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim


# flexible MLP class
class LightningFlexNN(LightningModelBase):
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
        - elementwise_affine – a boolean value that when set to True, this module has learnable per-element affine
        parameters initialized to ones (for weights) and zeros (for biases). Default: True.

    MaxPool Layer Parameters:
        - kernel_size – the size of the window to take a max over
        - stride – the stride of the window. Default value is kernel_size
        - padding – implicit zero padding to be added on both sides
        - dilation – a parameter that controls the stride of elements in the window
        - return_indices – if True, will return the max indices along with the outputs. Useful for
        torch.nn.MaxUnpool2d later
        - ceil_mode – when True, will use ceil instead of floor to compute the output shape
    """

    def __init__(self, hparams: Namespace) -> None:
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
        self.get_functions()
        self.set_channels()
        self.min_val_loss: Optional[torch.Tensor] = None

        self.layers_list = []

        self.height = self.hparams.height
        self.width = self.hparams.width
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,)
        self.construct_nn2d(layer_list=self.hparams.layers)

        if hasattr(self.hparams, 'mlp_layer'):
            self.layers_list.append(torch.nn.Flatten())
            in_dim = self.final_channel * self.height * self.width
            self.construct_mlp(in_dim, self.hparams.mlp_layer['hidden_layer'], self.hparams.mlp_layer['n_out'])

        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers_list)

    def set_channels(self) -> None:
        in_channels = self.hparams.depth
        for i, layer_dict in enumerate(self.hparams.layers):
            if layer_dict['type'] == 'Conv2d' and all(
                    elem not in layer_dict for elem in ['in_channels', 'out_channels']):
                out_channels = layer_dict['params'].pop('channels')
                self.hparams.layers[i]['params']['in_channels'] = in_channels
                self.hparams.layers[i]['params']['out_channels'] = out_channels
                in_channels = out_channels
        self.final_channel = in_channels

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('depth', dtype=int, required=True)
        options['hparams'].add_key('width', dtype=int, required=True)
        options['hparams'].add_key('height', dtype=int, required=True)
        options['hparams'].add_key('layers', dtype=list, required=True)
        options['hparams'].add_key('mlp_layer', dtype=dict, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['layers'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'create_model', 'layers']))
        options['layers'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['layers'].add_key('params', dtype=dict, param_dict=True)

        options['mlp_layer'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'create_model',
                                                                                   'mlp_layer']))
        options['mlp_layer'].add_key('n_out', dtype=int, required=True)
        options['mlp_layer'].add_key('hidden_layer', dtype=list, required=True)

        options['optimizer'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: list) -> str:
        template = {'Model': {'type': 'LightningFlexNN',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'width': 'int', 'height': 'int', 'depth': 'int',
                                               'layers': [{'type': 'torch.nn module',
                                                           'params': {'module_param_1': 'value',
                                                                      'module_param_2': 'value'}},
                                                          {'type': 'e. g. Conv2d',
                                                           'params': {'kernel_size': 3, 'channels': 20}},
                                                          {'type': 'e. g. MaxPool2d', 'params': {'kernel_size': 2}}],
                                               'mlp_layer': {'n_out': 'int', 'hidden_layer': ['int', 'int', '...']},
                                               'output_activation': 'str (default: None)',
                                               'activation': 'str (default: ReLU)'},
                              'params': {'loss': 'str (default:MSELoss)',
                                         'optimizer': {'type': 'str (default: Adam)',
                                                       'params': {'lr': 'float (default: 1.e-3'}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'cooldown': 'int', 'patience': 'int',
                                                                  'min_lr': 'float'}},
                                         'num_workers': 'int (default: 10)', 'batch': 'int (default: 64)'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
