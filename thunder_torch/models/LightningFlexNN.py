#######################################################################################################################
# flexible CNN model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import Optional, List

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
import thunder_torch as tt


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
        self.min_val_loss: Optional[torch.Tensor] = None

        if hasattr(self.hparams, 'layers'):
            # if only a single layer is given, transform it to list object
            if not isinstance(self.hparams.layers, list):
                self.hparams.layers = [self.hparams.layers]

            self.hparams.layers, self.final_channel = self.set_channels(self.hparams.start_channels,
                                                                        self.hparams.layers)

            in_dim = None
            if self.hparams.layers[0]['type'] == 'Conv1d':
                self.height = self.hparams.height
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.layers[0]['type'] == 'Conv2d':
                self.height = self.hparams.height
                self.width = self.hparams.width
                self.construct_nn2d(layer_list=self.hparams.layers)
                in_dim = self.final_channel * self.height * self.width

            elif self.hparams.layers[0]['type'] == 'Conv3d':
                self.height = self.hparams.height
                self.width = self.hparams.width
                self.depth = self.hparams.depth
                self.construct_nn3d(layer_list=self.hparams.layers)
                in_dim = self.final_channel * self.height * self.width * self.depth

            elif self.hparams.layers[0]['type'] == 'Linear':
                raise NotImplementedError('Support for an MLP layer as starting layer in the layers dict not '
                                          'implemented yet. If an only mlp network should be constructed, pls use the '
                                          'key "mlp_layer" to construct it or the LightningFlexMLP module!')

            else:
                raise KeyError(f'Type "{self.hparams.layers[0]["type"]}" as starting layer for the network is not '
                               f'supported. Please start with either an convolutional or linear layer!')

        if hasattr(self.hparams, 'mlp_layer'):

            if in_dim is not None and 'n_in' in self.hparams.mlp_layer:
                assert in_dim == self.hparams.mlp_layer['n_in'], 'Entered input dimension of MLP Network not equal ' \
                                                                   'to the one calculated '
            elif in_dim is None and 'n_in' not in self.hparams.mlp_layer:
                raise KeyError('Input dimension of MLP network is missing, please add "n_in" key to "mlp_layer" dict')
            elif in_dim is not None and 'n_in' not in self.hparams.mlp_layer:
                self.hparams.mlp_layer['n_in'] = in_dim

            self.layers_list.append(torch.nn.Flatten())  # type: ignore[attr-defined]
            self.construct_mlp(self.hparams.mlp_layer['n_in'], self.hparams.mlp_layer['hidden_layer'],
                               self.hparams.mlp_layer['n_out'])

        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers_list)

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexNN.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('height', dtype=int, required=True)
        options['hparams'].add_key('width', dtype=int)  # only required for 2d layers
        options['hparams'].add_key('depth', dtype=int)  # only required for 3d layers
        options['hparams'].add_key('start_channels', dtype=int)
        options['hparams'].add_key('layers', dtype=list)
        options['hparams'].add_key('mlp_layer', dtype=dict)
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
        options['mlp_layer'].add_key('n_in', dtype=int)
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
    def yaml_template(key_list: List[str]) -> str:
        template = {'Model': {'type': 'LightningFlexNN',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'height': 'int',
                                               'width': 'int (only for 2d models)',
                                               'depth': 'int (only for 3d models)',
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

        template = tt.utils.get_by_path(template, key_list)

        return yaml.dump(template, sort_keys=False)
