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
class LightningFlexAutoEncoderMultiInput(LightningModelBase):
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

        # add hparams keyword so that model can be easly restored (see utils/general.py::load_model_from_checkpoint)
        self.hparams.model_type = 'LightningFlexAutoEncoderMultiTimeStep'

        # Encoder Layers for the single inputs ########################################################################
        if hasattr(self.hparams, 'encoder_single'):
            self.applied_encoder_single = True
            self.hparams.encoder_single['layers'], self.final_channel = self.set_channels(
                self.hparams.input_dim['start_channels'],
                self.hparams.encoder_single['layers'])

            if self.hparams.encoder_single['layers'][0]['type'] == 'Conv1d':
                self.height = self.hparams.input_dim['height']
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.encoder_single['layers'][0]['type'] == 'Conv2d':
                self.height = self.hparams.input_dim['height']
                self.width = self.hparams.input_dim['width']
                self.construct_nn2d(layer_list=self.hparams.encoder_single['layers'])

            elif self.hparams.encoder_single['layers'][0]['type'] == 'Conv3d':
                self.height = self.hparams.input_dim['height']
                self.width = self.hparams.input_dim['width']
                self.depth = self.hparams.input_dim['depth']
                self.construct_nn3d(layer_list=self.hparams.encoder_single['layers'])

            self.encoder_single = [torch.nn.Sequential(*self.layers_list) for _ in
                                   range(self.hparams.input_dim['multi_input'])]

            # reset layers list for combined encoder, starting with the cat layer
            self.layers_list = []
            self.layers_list.append(Cat(dim=self.hparams.encoder_single['encoder_config']['cat_dim']))

        else:
            self.applied_encoder_single = False
            _logger.info('No definition of an encoder for the single input is given (key: "encoder_single"), so that '
                         'only a single input is used. Encoders for multiple inputs can be set either individual for '
                         'the input (-> enter a list of structures) or identical for every input (-> enter single '
                         'structure)')

        # Encoder Layers for the combined input ########################################################################
        if hasattr(self.hparams, 'encoder_combined'):
            if self.applied_encoder_single:
                self.hparams.encoder_combined, self.final_channel = self.set_channels(self.final_channel,
                                                                                      self.hparams.encoder_combined)

            else:
                self.hparams.encoder_combined, self.final_channel = self.set_channels(
                    self.hparams.input_dim['start_channels'],
                    self.hparams.encoder_combined)

            if self.hparams.encoder_combined[0]['type'] == 'Conv1d':
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.encoder_combined[0]['type'] == 'Conv2d':
                self.construct_nn2d(layer_list=self.hparams.encoder_combined)

            elif self.hparams.encoder_combined[0]['type'] == 'Conv3d':
                self.construct_nn3d(layer_list=self.hparams.encoder_combined)
        else:
            raise KeyError('For this particular network, the definition of an encoder for the combined input is '
                           'necessary (key: "encoder_combined")')

        # Possible MLP between De and Encoder ##########################################################################
        if hasattr(self.hparams, 'mlp_layer'):
            self.applied_mlp = True
            # check if encoder layers already applied, if yes apply Flatten operation
            self.layers_list.append(torch.nn.Flatten())  # type: ignore[attr-defined]
            in_dim = self.final_channel * self.height * self.width

            if 'n_out' in self.hparams.mlp_layer:
                out_dim = self.hparams.mlp_layer['n_out']
            else:
                out_dim = in_dim

            self.construct_mlp(in_dim, self.hparams.mlp_layer['hidden_layer'], out_dim)
            self.layers_list.append(self.activation_fn)
        else:
            self.applied_mlp = False
            _logger.info('No MLP between De and Encoder')

        # Decoder Layers for the combined input ########################################################################
        if hasattr(self.hparams, 'decoder_combined'):
            # check if encoder and/or mlp layers already
            self.hparams.decoder_combined, self.final_channel = self.set_channels(self.final_channel,
                                                                                  self.hparams.decoder_combined)

            if self.hparams.decoder_combined[0]['type'] == 'Conv1Transposed':
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.decoder_combined[0]['type'] == 'ConvTranspose2d':
                if self.applied_mlp:
                    self.layers_list.append(Reshape([self.height, self.width],
                                                    self.hparams.decoder_combined[0]['params']['in_channels']))
                self.construct_nn2d(layer_list=self.hparams.decoder_combined)

            elif self.hparams.decoder_combined[0]['type'] == 'ConvTranspose3d':
                if self.applied_mlp:
                    self.layers_list.append(Reshape([self.height, self.width, self.depth],
                                                    self.hparams.decoder_combined[0]['params']['in_channels']))
                self.construct_nn3d(layer_list=self.hparams.decoder_combined)

            elif self.hparams.decoder[0]['type'] == 'Upsample':
                # TODO implement upsample method
                raise NotImplementedError('Support for Upsample not implemented yet')
        else:
            raise KeyError('For this particular network, the definition of an decoder for the combined input is '
                           'necessary (key: "decoder_combined")')

        # Decoder Layers for the single outputs ########################################################################
        if hasattr(self.hparams, 'decoder_single'):
            # add split layer at the end of the combined decoder to get the single outputs for the single decoder
            self.layers_list.append(Split(self.hparams.decoder_single['decoder_config']['split_size_or_sections'],
                                          dim=self.hparams.decoder_single['decoder_config']['split_dim']))
            self.layers = torch.nn.Sequential(*self.layers_list)
            self.layers_list = []

            self.hparams.decoder_single['layers'], self.final_channel = \
                self.set_channels(self.final_channel, self.hparams.decoder_single['layers'])

            if self.hparams.decoder_single['layers'][0]['type'] == 'Conv1Transposed':
                assert NotImplementedError('Support for 1d Conv layers not implemented at the moment')
                # TODO: implement support

            elif self.hparams.decoder_single['layers'][0]['type'] == 'ConvTranspose2d':
                self.construct_nn2d(layer_list=self.hparams.decoder_single['layers'])

            elif self.hparams.decoder_single['layers'][0]['type'] == 'ConvTranspose3d':
                self.construct_nn3d(layer_list=self.hparams.decoder_single['layers'])

            elif self.hparams.decoder_single['layers'][0]['type'] == 'Upsample':
                # TODO implement upsample method
                raise NotImplementedError('Support for Upsample not implemented yet')

            self.decoder_single = [torch.nn.Sequential(*self.layers_list) for _ in
                                   range(self.hparams.input_dim['multi_output'])]
        else:
            self.layers = torch.nn.Sequential(*self.layers_list)
            _logger.info('Network has no separation in different outputs with individual decoder layers')

        _logger.info('Model build succeed!')

    def set_channels(self, in_channels: int, layer_dicts: List[dict]) -> Tuple[List[dict], int]:
        for i, layer_dict in enumerate(layer_dicts):  # TODO: let automatically adapt for all conv layers
            if any(layer_dict['type'] == item for item in self.channel_computation) \
                    and all(elem not in layer_dict['params'] for elem in ['in_channels', 'out_channels']):
                out_channels = layer_dict['params'].pop('channels')
                layer_dicts[i]['params']['in_channels'] = in_channels
                layer_dicts[i]['params']['out_channels'] = out_channels
                in_channels = out_channels
            elif layer_dict['type'] == 'Conv3d' and all(elem in layer_dict['params']
                                                        for elem in ['in_channels', 'out_channels']):
                in_channels = layer_dicts[i]['params']['out_channels']

        return layer_dicts, in_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        forward pass through the network

        Parameters
        ----------
        x           - input tensor of the pytorch.nn.Linear layer

        Returns
        -------
        x           - output tensor of the pytorch.nn.Linear layer
        """
        input_list = [self.encoder_single[i](x[:, i, :, :, :, :]) for i in range(x.shape[1])]
        output_list = self.layers(input_list)
        x = [self.decoder_single[i](output_list[i]) for i in range(len(output_list))]
        return x

    def configure_optimizers(self) -> Union[object, tuple]:
        """
        optimizer and lr scheduler

        Returns
        -------
        optimizer       - PyTorch Optimizer function
        scheduler       - PyTorch Scheduler function
        """
        for m in _modules_optim:
            try:
                _, optimizer_cls = dynamic_imp(m, self.hparams.optimizer['type'])
                # optimizer_cls = getattr(importlib.import_module(m), self.hparams.optimizer['type'])
                break
            except AttributeError or ModuleNotFoundError:
                _logger.debug('Optimizer of type {} NOT found in {}'.format(self.hparams.optimizer['type'], m))

        params = []
        if hasattr(self, 'encoder_single'):
            for i in range(len(self.encoder_single)):
                params += list(self.encoder_single[i].parameters())
        params += list(self.layers.parameters())
        if hasattr(self, 'decoder_single'):
            for i in range(len(self.decoder_single)):
                params += list(self.decoder_single[i].parameters())

        try:
            if 'params' in self.hparams.optimizer:
                optimizer = optimizer_cls(params, **self.hparams.optimizer['params'])
            else:
                optimizer = optimizer_cls(params)
        except NameError:
            raise NameError(f'Optimizer "{self.hparams.optimizer["type"]}" cannot be found in given '
                            f'sources: "{_modules_optim}"')

        if self.hparams.scheduler['execute']:
            for m in _modules_lr_scheduler:
                try:
                    _, scheduler_cls = dynamic_imp(m, self.hparams.scheduler['type'])
                    scheduler = scheduler_cls(optimizer, **self.hparams.scheduler['params'])
                    # scheduler = getattr(importlib.import_module(m), self.hparams.scheduler['type'])\
                    #     (optimizer, **self.hparams.scheduler['params'])
                    break
                except AttributeError or ModuleNotFoundError:
                    _logger.debug('LR Scheduler of type {} not found in {}'.format(self.hparams.scheduler['type'], m))

            try:
                return [optimizer], [scheduler]
            except NameError:
                raise NameError(f'LR Scheduler "{self.hparams.scheduler["type"]}" cannot be found in given '
                                f'sources: "{_modules_lr_scheduler}"')

        else:
            return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Training step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.zeros(1)
        for i in range(len(y_hat)):
            loss += self.loss_fn(y_hat[i], y[:, i, :, :, :, :])
        log = {'train_loss': loss}
        hiddens = {'inputs': x[:, 1, :, :, :, :].detach(), 'preds': y_hat[1].detach(),
                   'targets': y[:, 1, :, :, :, :].detach()}  # TODO general solution
        results = {'loss': loss, 'log': log, 'hiddens': hiddens}
        return results

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Validation step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.zeros(1)
        for i in range(len(y_hat)):
            loss += self.loss_fn(y_hat[i], y[:, i, :, :, :, :])
        hiddens = {'inputs': x[:, 1, :, :, :, :].detach(), 'preds': y_hat[1].detach(),
                   'targets': y[:, 1, :, :, :, :].detach()}  # TODO general solution
        return {'val_loss': loss, 'hiddens': hiddens}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.zeros(1)
        for i in range(len(y_hat)):
            loss += self.loss_fn(y_hat[i], y[:, i, :, :, :, :])
        hiddens = {'inputs': x[:, 1, :, :, :, :].detach(), 'preds': y_hat[1].detach(),
                   'targets': y[:, 1, :, :, :, :].detach()}  # TODO general solution
        return {'test_loss': loss, 'hiddens': hiddens}

    @staticmethod
    def get_OptionClass() -> dict:
        options = {'hparams': OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                          yaml_template(['Model', 'params']))}
        options['hparams'].add_key('input_dim', dtype=[int, dict], required=True)
        options['hparams'].add_key('encoder_single', dtype=dict)
        options['hparams'].add_key('encoder_combined', dtype=list, required=True)
        options['hparams'].add_key('mlp_layer', dtype=dict)
        options['hparams'].add_key('decoder_combined', dtype=list, required=True)
        options['hparams'].add_key('decoder_single', dtype=dict)
        options['hparams'].add_key('output_activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('activation', dtype=str, attr_of=_modules_activation)
        options['hparams'].add_key('loss', dtype=str, attr_of=_modules_loss)
        options['hparams'].add_key('optimizer', dtype=dict)
        options['hparams'].add_key('scheduler', dtype=dict)
        options['hparams'].add_key('num_workers', dtype=int)
        options['hparams'].add_key('batch', dtype=int)
        options['hparams'].add_key('lparams', dtype=Namespace)

        options['input_dim'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                           yaml_template(['Model', 'create_model', 'input_dim']))
        options['input_dim'].add_key('height', dtype=int, required=True)
        options['input_dim'].add_key('width', dtype=int, required=True)
        options['input_dim'].add_key('depth', dtype=int)  # only required for 3d Conv layers
        options['input_dim'].add_key('start_channels', dtype=int, required=True)
        options['input_dim'].add_key('multi_input', dtype=int)
        options['input_dim'].add_key('multi_output', dtype=int)

        options['encoder_combined'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                  yaml_template(['Model', 'create_model', 'encoder_combined']))
        options['encoder_combined'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['encoder_combined'].add_key('params', dtype=dict, param_dict=True)

        options['encoder_single'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                yaml_template(['Model', 'create_model', 'encoder_single']))
        options['encoder_single'].add_key('encoder_config', dtype=dict, required=True)
        options['encoder_single'].add_key('layers', dtype=list, required=True)

        options['encoder_config'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                yaml_template(['Model', 'create_model', 'encoder_single',
                                                               'encoder_config']))
        options['encoder_config'].add_key('cat_dim', dtype=int, required=True)

        options['decoder_combined'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                  yaml_template(['Model', 'create_model', 'decoder_combined']))
        options['decoder_combined'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['decoder_combined'].add_key('params', dtype=dict, param_dict=True)

        options['decoder_single'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                yaml_template(['Model', 'create_model', 'decoder_single']))
        options['decoder_single'].add_key('decoder_config', dtype=dict, required=True)
        options['decoder_single'].add_key('layers', dtype=list, required=True)

        options['decoder_config'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                                yaml_template(['Model', 'create_model', 'decoder_single',
                                                               'decoder_config']))
        options['decoder_config'].add_key('split_dim', dtype=int, required=True)
        options['decoder_config'].add_key('split_size_or_sections', dtype=[int, list], required=True)

        options['layers'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                        yaml_template(['Model', 'create_model', 'encoder_single', 'layers']))
        options['layers'].add_key('type', dtype=str, required=True, attr_of='torch.nn')
        options['layers'].add_key('params', dtype=dict, param_dict=True)

        options['mlp_layer'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                           yaml_template(['Model', 'create_model', 'mlp_layer']))
        options['mlp_layer'].add_key('n_out', dtype=int)
        options['mlp_layer'].add_key('hidden_layer', dtype=list, required=True)

        options['optimizer'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                           yaml_template(['Model', 'params', 'optimizer']))
        options['optimizer'].add_key('type', dtype=str, attr_of=_modules_optim)
        options['optimizer'].add_key('params', dtype=dict, param_dict=True)

        options['scheduler'] = OptionClass(template=LightningFlexAutoEncoderMultiInput.
                                           yaml_template(['Model', 'params', 'scheduler']))
        options['scheduler'].add_key('execute', dtype=bool)
        options['scheduler'].add_key('type', dtype=str, attr_of=_modules_lr_scheduler)
        options['scheduler'].add_key('params', dtype=dict, param_dict=True)

        return options

    @staticmethod
    def yaml_template(key_list: List[str]) -> str:
        template = {'Model': {'type': 'LightningFlexAutoEncoderMultiTimeStep',
                              '###INFO###': 'load_model and create_model are mutually exclusive',
                              'load_model': {'path': 'name.ckpt'},
                              'create_model': {'input_dim': {'width': 'int', 'height': 'int',
                                                             'depth': 'int (only for 3d models)',
                                                             'start_channels': 'int',
                                                             'multi_input': 'int',
                                                             'multi_output': 'int'},
                                               'encoder_single': {'encoder_config': {'cat_dim': 'int'},
                                                                  'layers': [{'type': 'torch.nn module',
                                                                              'params': {'module_param_1': 'value',
                                                                                         'module_param_2': 'value'}},
                                                                             {'type': 'e. g. Conv2d',
                                                                              'params': {'kernel_size': 3,
                                                                                         'channels': 20}},
                                                                             {'type': 'e. g. MaxPool2d',
                                                                              'params': {'kernel_size': 2}}]},
                                               'encoder_combined': [{'type': 'torch.nn module',
                                                                     'params': {'module_param_1': 'value',
                                                                                'module_param_2': 'value'}},
                                                                    {'type': 'e. g. Conv2d',
                                                                     'params': {'kernel_size': 3, 'channels': 20}},
                                                                    {'type': 'e. g. MaxPool2d',
                                                                     'params': {'kernel_size': 2}}],
                                               'mlp_layer': {'n_out': 'int', 'hidden_layer': ['int', 'int', '...']},
                                               'decoder_combined': [{'type': 'torch.nn module',
                                                                     'params': {'module_param_1': 'value',
                                                                                'module_param_2': 'value'}},
                                                                    {'type': 'e. g. ConvTranspose2d',
                                                                     'params': {'kernel_size': 3, 'channels': 20}},
                                                                    {'type': 'e. g. MaxUnpool2d',
                                                                     'params': {'kernel_size': 2}}],
                                               'decoder_single': {'decoder_config': {'split_dim': 'int',
                                                                                     'split_size_or_sections':
                                                                                         'int or List[int]'},
                                                                  'layers': [{'type': 'torch.nn module',
                                                                              'params': {'module_param_1': 'value',
                                                                                         'module_param_2': 'value'}},
                                                                             {'type': 'e. g. ConvTranspose2d',
                                                                              'params': {'kernel_size': 3,
                                                                                         'channels': 20}},
                                                                             {'type': 'e. g. MaxUnpool2d',
                                                                              'params': {'kernel_size': 2}}]},
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
