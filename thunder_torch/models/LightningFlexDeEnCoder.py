#######################################################################################################################
# flexible De-Encoder network using PyTorch and PyTorch Lightning
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import List, Union, Optional

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
from thunder_torch.models._utils import Reshape
from thunder_torch import _logger
import thunder_torch as tt


# flexible MLP class
class LightningFlexDeEnCoder(LightningModelBase):
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

        in_encoder_dim = None
        out_encoder_dim = None

        if hasattr(self.hparams, 'cnn_encoder'):
            self.cnn_encoder_applied = True
            self.hparams.cnn_encoder, self.final_channel = self.set_channels(self.hparams.input_dim['start_channels'],
                                                                             self.hparams.cnn_encoder)

            if self.hparams.cnn_encoder[0]['type'] == 'Conv1d':
                self.height = self.hparams.input_dim['height']
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.cnn_encoder[0]['type'] == 'Conv2d':
                self.height = self.hparams.input_dim['height']
                self.width = self.hparams.input_dim['width']
                self.construct_nn2d(layer_list=self.hparams.cnn_encoder)
                in_encoder_dim = self.final_channel * self.height * self.width

            elif self.hparams.cnn_encoder[0]['type'] == 'Conv3d':
                self.height = self.hparams.input_dim['height']
                self.width = self.hparams.input_dim['width']
                self.depth = self.hparams.input_dim['depth']
                self.construct_nn3d(layer_list=self.hparams.cnn_encoder)
                in_encoder_dim = self.final_channel * self.height * self.width * self.depth

        else:
            self.cnn_encoder_applied = False

        if hasattr(self.hparams, 'mlp_encoder'):
            self.mlp_encoder_applied = True

            # check if encoder layers already applied, if yes apply Flatten operation
            if self.cnn_encoder_applied:
                self.layers_list.append(torch.nn.Flatten())  # type: ignore[attr-defined]
            else:
                in_encoder_dim = self.hparams.input_dim

            out_encoder_dim = self.hparams.mlp_encoder['hidden_layer'][-1]

            self.construct_mlp(in_encoder_dim, self.hparams.mlp_encoder['hidden_layer'][:-1], out_encoder_dim)
            self.layers_list.append(self.activation_fn)
        else:
            self.mlp_encoder_applied = False

        # summarize layers upto this point as encoder
        if self.layers_list:
            self.encoder = torch.nn.Sequential(*self.layers_list)
            self.layers_list = []

        if hasattr(self.hparams, 'mlp_decoder'):
            self.mlp_decoder_applied = True

            if self.mlp_encoder_applied:
                in_decoder_dim = out_encoder_dim
                out_decoder_dim = in_encoder_dim
                hiddens = self.hparams.mlp_decoder['hidden_layer']
            else:
                in_decoder_dim = self.hparams.input_dim
                out_decoder_dim = self.hparams.mlp_decoder['hidden_layer'][-1]
                hiddens = self.hparams.mlp_decoder['hidden_layer'][:-1]

            self.construct_mlp(in_decoder_dim, hiddens, out_decoder_dim)
            self.layers_list.append(self.activation_fn)

        else:
            self.mlp_decoder_applied = False

        if hasattr(self.hparams, 'cnn_decoder'):
            self.cnn_decoder_applied = True

            # check if encoder and/or mlp layers already
            if self.cnn_encoder_applied:
                self.hparams.cnn_decoder, self.final_channel = self.set_channels(
                    self.final_channel, self.hparams.cnn_decoder)
            else:
                self.hparams.cnn_decoder, self.final_channel = self.set_channels(
                    self.hparams.input_dim['start_channels'], self.hparams.cnn_decoder)

            if self.hparams.cnn_decoder[0]['type'] == 'Conv1Transposed':
                self.height = self.hparams.input_dim['height']
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.cnn_decoder[0]['type'] == 'ConvTranspose2d':
                if not self.cnn_encoder_applied:
                    self.height = self.hparams.input_dim['height']
                    self.width = self.hparams.input_dim['width']
                if self.mlp_decoder_applied or self.mlp_encoder_applied:  # TODO: what if no cnn preior defined
                    self.layers_list.append(Reshape([self.height, self.width],
                                                    self.hparams.cnn_decoder[0]['params']['in_channels']))
                self.construct_nn2d(layer_list=self.hparams.cnn_decoder)

            elif self.hparams.cnn_decoder[0]['type'] == 'ConvTranspose3d':
                if not self.cnn_encoder_applied:
                    self.height = self.hparams.input_dim['height']
                    self.width = self.hparams.input_dim['width']
                    self.depth = self.hparams.input_dim['depth']
                if self.mlp_decoder_applied or self.mlp_encoder_applied:
                    self.layers_list.append(Reshape([self.height, self.width, self.depth],
                                                    self.hparams.cnn_decoder[0]['params']['in_channels']))
                self.construct_nn3d(layer_list=self.hparams.cnn_decoder)

            elif self.hparams.cnn_decoder[0]['type'] == 'Upsample':
                # TODO implement upsample method
                raise NotImplementedError('Support for Upsample not implemented yet')

        if self.layers_list:
            self.decoder = torch.nn.Sequential(*self.layers_list)
            self.layers_list = []

    def get_optimizer_parameters(self) -> Union[torch.Generator, List[torch.Generator]]:
        # define model parameters which should be optimized
        params = []
        params += list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        _logger.debug('Encoder and Decoder parameters selected to be optimized')
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'params']))}
        options['hparams'].add_key('input_dim', dtype=[int, dict], required=True)
        options['hparams'].add_key('cnn_encoder', dtype=list)
        options['hparams'].add_key('mlp_encoder', dtype=dict)
        options['hparams'].add_key('mlp_decoder', dtype=dict)
        options['hparams'].add_key('cnn_decoder', dtype=list)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['input_dim'] = OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'create_model',
                                                                                          'input_dim']))
        options['input_dim'].add_key('height', dtype=int, required=True)
        options['input_dim'].add_key('width', dtype=int, required=True)
        options['input_dim'].add_key('depth', dtype=int)  # only required for 3d Conv layers
        options['input_dim'].add_key('start_channels', dtype=int, required=True)

        options['cnn_encoder'] = OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'create_model',
                                                                                        'cnn_encoder']))
        options['cnn_encoder'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['cnn_encoder'].add_key('params', dtype=dict, param_dict=True)
        options['cnn_encoder'].add_key('activation', dtype=bool)
        options['cnn_decoder'] = options['cnn_encoder']

        options['mlp_encoder'] = OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'create_model',
                                                                                            'mlp_encoder']))
        options['mlp_encoder'].add_key('hidden_layer', dtype=list, required=True)
        options['mlp_decoder'] = options['mlp_encoder']

        options['optimizer'] = OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'params',
                                                                                          'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexDeEnCoder.yaml_template(['Model', 'params',
                                                                                          'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: List[str]) -> str:
        template = {'Model': {'type': 'LightningFlexDeEnCoder',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'input_dim': {'width': 'int', 'height': 'int',
                                                             'depth': 'int (only for 3d models)',
                                                             'start_channels': 'int'},
                                               'cnn_encoder': [{'type': 'torch.nn module',
                                                                'params': {'module_param_1': 'value',
                                                                           'module_param_2': 'value'}},
                                                               {'type': 'e. g. Conv2d',
                                                                'params': {'kernel_size': 3, 'channels': 20}},
                                                               {'type': 'e. g. MaxPool2d',
                                                                'params': {'kernel_size': 2}}],
                                               'mlp_encoder': {'hidden_layer': ['int', 'int', '...']},
                                               'mlp_decoder': {'hidden_layer': ['int', 'int', '...']},
                                               'cnn_decoder': [{'type': 'torch.nn module',
                                                                'params': {'module_param_1': 'value',
                                                                           'module_param_2': 'value'}},
                                                               {'type': 'e. g. ConvTranspose2d',
                                                                'params': {'kernel_size': 3, 'channels': 20}},
                                                               {'type': 'e. g. MaxUnpool2d',
                                                                'params': {'kernel_size': 2}}],
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
