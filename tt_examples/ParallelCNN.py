#######################################################################################################################
# flexible De-Encoder network using PyTorch and PyTorch Lightning
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import List, Optional, Union, Tuple

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
from thunder_torch.models._utils import Reshape, Cat, Split
import thunder_torch as tt
from thunder_torch import _logger
from thunder_torch.utils.general import dynamic_imp


# flexible MLP class
class LightningFlexCNNParallel(LightningModelBase):
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
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,
                                 torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,)
        self.channel_computation = ['Conv1d', 'Conv1d', 'Conv3d', 'ConvTranspose1d', 'ConTranspose2d',
                                    'ConvTranspose3d']

        self.width: int
        self.height: int
        self.depth: int

        # Layers 1 ####################################################################################################
        self.layers_list = []
        self.hparams.layers1, self.final_channel = self.set_channels(self.hparams.input_dim['start_channels'],
                                                                     self.hparams.layers1)
        self.get_layers(self.hparams.layers1)
        self.layers_list.append(torch.nn.Flatten())
        self.layers1 = torch.nn.Sequential(*self.layers_list)

        # Layers 2 ####################################################################################################
        self.layers_list = []
        self.hparams.layers2, self.final_channel = self.set_channels(self.hparams.input_dim['start_channels'],
                                                                     self.hparams.layers2)
        self.get_layers(self.hparams.layers2)
        self.layers_list.append(torch.nn.Flatten())
        self.layers2 = torch.nn.Sequential(*self.layers_list)

        # Layers 3 ####################################################################################################
        self.layers_list = []
        self.hparams.layers3, self.final_channel = self.set_channels(self.hparams.input_dim['start_channels'],
                                                                     self.hparams.layers3)
        self.get_layers(self.hparams.layers3)
        self.layers_list.append(torch.nn.Flatten())
        self.layers3 = torch.nn.Sequential(*self.layers_list)

        # Layers MLP ##################################################################################################
        self.layers_list = []
        self.construct_mlp(self.hparams.mlp_layer['n_in'], self.hparams.mlp_layer['hidden_layer'],
                           self.hparams.mlp_layer['n_out'])
        self.layersMLP = torch.nn.Sequential(*self.layers_list)

        _logger.info('Model build succeed!')

    def get_layers(self, layer_list: List):
        if self.hparams.layers2[0]['type'] == 'Conv1d':
            self.height = self.hparams.input_dim['height']
            assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
            # TODO: implement support

        elif self.hparams.layers2[0]['type'] == 'Conv2d':
            self.height = self.hparams.input_dim['height']
            self.width = self.hparams.input_dim['width']
            self.construct_nn2d(layer_list=layer_list)

        elif self.hparams.layers2[0]['type'] == 'Conv3d':
            self.height = self.hparams.input_dim['height']
            self.width = self.hparams.input_dim['width']
            self.depth = self.hparams.input_dim['depth']
            self.construct_nn3d(layer_list=layer_list)

    def get_optimizer_parameters(self) -> Union[torch.Generator, List[torch.Generator]]:
        params = []
        params += list(self.layers1.parameters())
        params += list(self.layers2.parameters())
        params += list(self.layers3.parameters())
        params += list(self.layersMLP.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layers1(x)
        x2 = self.layers2(x)
        x3 = self.layers3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.layersMLP(x)
        x = torch.squeeze(x)
        return x

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexCNNParallel.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('input_dim', dtype=[int, dict], required=True)
        options['hparams'].add_key('layers1', dtype=list, required=True)
        options['hparams'].add_key('layers2', dtype=list, required=True)
        options['hparams'].add_key('layers3', dtype=list, required=True)
        options['hparams'].add_key('mlp_layer', dtype=dict)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['input_dim'] = OptionClass(template=LightningFlexCNNParallel.
                                           yaml_template(['Model', 'create_model', 'input_dim']))
        options['input_dim'].add_key('height', dtype=int, required=True)
        options['input_dim'].add_key('width', dtype=int, required=True)
        options['input_dim'].add_key('depth', dtype=int)  # only required for 3d Conv layers
        options['input_dim'].add_key('start_channels', dtype=int, required=True)

        options['layers1'] = OptionClass(template=LightningFlexCNNParallel.
                                         yaml_template(['Model', 'create_model', 'layers1']))
        options['layers1'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['layers1'].add_key('params', dtype=dict, param_dict=True)

        options['layers2'] = options['layers1']
        options['layers3'] = options['layers1']

        options['mlp_layer'] = OptionClass(template=LightningFlexCNNParallel.
                                           yaml_template(['Model', 'create_model', 'mlp_layer']))
        options['mlp_layer'].add_key('n_out', dtype=int)
        options['mlp_layer'].add_key('hidden_layer', dtype=list, required=True)
        options['mlp_layer'].add_key('n_in', dtype=int, required=True)

        options['optimizer'] = OptionClass(template=LightningFlexCNNParallel.
                                           yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexCNNParallel.
                                           yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: List[str]) -> str:
        template = {'Model': {'type': 'LightningFlexCNNParallel',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'input_dim': {'width': 'int', 'height': 'int',
                                                             'depth': 'int (only for 3d models)',
                                                             'start_channels': 'int',
                                                             'multi_input': 'int',
                                                             'multi_output': 'int'},
                                               'layers1': [{'type': 'torch.nn module',
                                                            'params': {'module_param_1': 'value',
                                                                       'module_param_2': 'value'}},
                                                           {'type': 'e. g. Conv2d',
                                                            'params': {'kernel_size': 3, 'channels': 20}},
                                                           {'type': 'e. g. MaxPool2d',
                                                            'params': {'kernel_size': 2}}],
                                               'mlp_layer': {'n_out': 'int', 'hidden_layer': ['int', 'int', '...'],
                                                             'n_in': 'int'},
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
