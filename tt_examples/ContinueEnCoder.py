#######################################################################################################################
# flexible De-Encoder network using PyTorch and PyTorch Lightning
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import List, Optional

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch.utils.general import load_model_from_checkpoint
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
from thunder_torch import _logger
import thunder_torch as tt


# flexible MLP class
class LightningFlexContinueModel(LightningModelBase):
    """
    Create a flex De-EnCoder network as PyTorch LightningModule

    The De- and Encoder can be any kind of convolutional layer while inbetween a MLP can be implemented.
    The implementation of each part (e.g. Encoder, MLP, Decoder) is optional and depends on the given parameters.
    However, it is crucial and quite logical that at least one of them has to be defined.

    In the following, parameters of the different layers of the De and Encoder can be found:

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
        Initializes a flexDeEnCoder model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters (encoder, mlp, decoder, lr, activation fn, ...)
        """
        super().__init__()

        self.hparams = hparams
        self.check_hparams()
        self.get_default()
        self.get_functions()
        self.min_val_loss: Optional[torch.Tensor] = None
        in_dim = None

        self.autoencoder = load_model_from_checkpoint(hparams.base_model['path'])
        if hasattr(self.hparams, 'freeze') and self.hparams.freeze:
            self.autoencoder.freeze()

        if 'layers' in self.hparams.add_model:
            # if only a single layer is given, transform it to list object
            if not isinstance(self.hparams.add_model['layers'], list):
                self.hparams.add_model['layers'] = [self.hparams.add_model['layers']]

            self.hparams.add_model['layers'], self.final_channel = self.set_channels(
                self.hparams.add_model['start_channels'], self.hparams.add_model['layers'])

            if self.hparams.add_model['layers'][0]['type'] == 'Conv1d':
                self.height = self.hparams.add_model['height']
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.add_model['layers'][0]['type'] == 'Conv2d':
                self.height = self.hparams.add_model['height']
                self.width = self.hparams.add_model['width']
                self.construct_nn2d(layer_list=self.hparams.add_model['layers'])
                in_dim = self.final_channel * self.height * self.width

            elif self.hparams.add_model['layers'][0]['type'] == 'Conv3d':
                self.height = self.hparams.add_model['height']
                self.width = self.hparams.add_model['width']
                self.depth = self.hparams.add_model['depth']
                self.construct_nn3d(layer_list=self.hparams.add_model['layers'])
                in_dim = self.final_channel * self.height * self.width * self.depth

            elif self.hparams.add_model['layers'][0]['type'] == 'Linear':
                raise NotImplementedError('Support for an MLP layer as starting layer in the layers dict not '
                                          'implemented yet. If an only mlp network should be constructed, pls use the '
                                          'key "mlp_layer" to construct it or the LightningFlexMLP module!')

            else:
                raise KeyError(f'Type "{self.hparams.layers[0]["type"]}" as starting layer for the network is not '
                               f'supported. Please start with either an convolutional or linear layer!')

        if 'mlp_layer' in self.hparams.add_model:

            if in_dim is not None and 'n_in' in self.hparams.add_model['mlp_layer']:
                assert in_dim == self.hparams.add_model['mlp_layer']['n_in'], 'Entered input dimension of MLP ' \
                                                                              'Network not equal to the one calculated'
            elif in_dim is None and 'n_in' not in self.hparams.add_model['mlp_layer']:
                raise KeyError('Input dimension of MLP network is missing, please add "n_in" key to "mlp_layer" dict')
            elif in_dim is not None and 'n_in' not in self.hparams.add_model['mlp_layer']:
                self.hparams.add_model['mlp_layer']['n_in'] = in_dim

            self.layers_list.append(torch.nn.Flatten())  # type: ignore[attr-defined]
            self.construct_mlp(self.hparams.add_model['mlp_layer']['n_in'],
                               self.hparams.add_model['mlp_layer']['hidden_layer'],
                               self.hparams.add_model['mlp_layer']['n_out'])

        if hasattr(self.hparams, 'output_activation'):
            self.layers_list.append(getattr(torch.nn, self.hparams.add_model['output_activation'])())

        self.layers = torch.nn.Sequential(*self.layers_list)
        _logger.info('Model build successful!')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.autoencoder.encoder(x)
        x = self.layers(x)
        x = torch.squeeze(x)
        return x

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexContinueModel.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('base_model', dtype=dict, required=True)
        options['hparams'].add_key('add_model', dtype=dict, required=True)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)
        options['hparams'].add_key('freeze', dtype=bool)

        options['base_model'] = OptionClass(template=LightningFlexContinueModel.yaml_template(
            ['Model', 'create_model', 'base_model']))
        options['base_model'].add_key('path', dtype=str)

        options['add_model'] = OptionClass(template=LightningFlexContinueModel.yaml_template(
            ['Model', 'create_model', 'add_model']))
        options['add_model'].add_key('height', dtype=int)
        options['add_model'].add_key('width', dtype=int)  # only required for 2d layers
        options['add_model'].add_key('depth', dtype=int)  # only required for 3d layers
        options['add_model'].add_key('start_channels', dtype=int)
        options['add_model'].add_key('layers', dtype=list)
        options['add_model'].add_key('mlp_layer', dtype=dict)

        options['layers'] = OptionClass(template=LightningFlexContinueModel.yaml_template(
            ['Model', 'create_model', 'add_model', 'layers']))
        options['layers'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['layers'].add_key('params', dtype=dict, param_dict=True)

        options['mlp_layer'] = OptionClass(template=LightningFlexContinueModel.yaml_template(
            ['Model', 'create_model', 'add_model', 'mlp_layer']))
        options['mlp_layer'].add_key('n_out', dtype=int, required=True)
        options['mlp_layer'].add_key('n_in', dtype=int)
        options['mlp_layer'].add_key('hidden_layer', dtype=list, required=True)

        options['optimizer'] = OptionClass(
            template=LightningFlexContinueModel.yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(
            template=LightningFlexContinueModel.yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: List[str]) -> str:
        template = {'Model': {'type': 'LightningFlexContinueModel',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'base_model': {'path': 'str'},
                                               'add_model': {'height': 'int',
                                                              'width': 'int (only for 2d models)',
                                                              'depth': 'int (only for 3d models)',
                                                              'layers': [{'type': 'torch.nn module',
                                                                          'params': {'module_param_1': 'value',
                                                                                     'module_param_2': 'value'}},
                                                                         {'type': 'e. g. Conv2d',
                                                                          'params': {'kernel_size': 3, 'channels': 20}},
                                                                         {'type': 'e. g. MaxPool2d',
                                                                          'params': {'kernel_size': 2}}],
                                                              'mlp_layer': {'n_out': 'int',
                                                                            'hidden_layer': ['int', 'int', '...']}},
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
