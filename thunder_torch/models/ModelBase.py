#######################################################################################################################
# Base for all models in the Toolbox, includes basic functions
#######################################################################################################################

# import packages
import torch
import pytorch_lightning as pl
from typing import List, Union, Tuple, Optional
from pathlib import Path
from collections.abc import Callable

from thunder_torch import _logger
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_optim, _modules_lr_scheduler
from thunder_torch.utils.general import dynamic_imp
from thunder_torch.models.ModelLayerConstuctors import LayerConstructors


# flexible MLP class
class LightningModelBase(pl.LightningModule, LayerConstructors):
    """
    Model Base of the Toolbox, includes repeating functions and functionalities to build network layers
    """
    def __init__(self) -> None:
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters (features, labels, lr, activation fn, ...)
        """
        super().__init__()

        self.loss_fn:       Callable[..., torch.Tensor]
        self.activation_fn: Callable[..., torch.Tensor]
        self.min_val_loss:  Optional[torch.Tensor] = None
        self.channel_computation = ['Conv1d', 'Conv1d', 'Conv3d', 'ConvTranspose1d', 'ConTranspose2d',
                                    'ConvTranspose3d']

        self.optimizer_parameters: Union[torch.Generator, List[torch.Generator]]

    def set_channels(self, in_channels: int, layer_dicts: List[dict]) -> Tuple[List[dict], int]:

        for i, layer_dict in enumerate(layer_dicts):
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

    def check_hparams(self) -> None:
        options = self.get_OptionClass()
        options["hparams"].add_key('model_type', dtype=str)
        OptionClass.checker(input_dict={'hparams': self.hparams}, option_classes=options)

    def get_default(self) -> None:
        if not self.hparams['optimizer']:
            self.hparams['optimizer']: dict = {'type': 'Adam', 'params': {'lr': 1e-3}}

        if not self.hparams['scheduler']:
            self.hparams['scheduler']: dict = {'execute': False}

        if 'model_type' not in self.hparams:
            class_name = str(self.__class__)
            class_name_split = class_name.split("'")[1]
            model_type = class_name_split.split(".")[-1]
            self.hparams['model_type'] = model_type

    def get_functions(self) -> None:
        """
        get activation and loss function from a source defined in _modules_activation or _modules_loss
        """
        for m in _modules_activation:
            try:
                _, activation_cls = dynamic_imp(m, self.hparams.activation)
                self.activation_fn = activation_cls()
                _logger.debug(f'{self.hparams.activation} fct found in {m}')
                if activation_cls:
                    break
                else:
                    continue
            except AttributeError or ModuleNotFoundError:
                _logger.debug(f'{self.hparams.activation} fct not found in {m}')
        assert self.activation_fn is not None, f'{self.hparams.activation} could not be found in {_modules_activation}'

        for m in _modules_loss:
            try:
                _, loss_cls = dynamic_imp(m, self.hparams.loss)
                self.loss_fn = loss_cls()
                # self.loss_fn = getattr(importlib.import_module(m), self.hparams.loss)()
                _logger.debug(f'{self.hparams.activation} fct found in {m}')
                if loss_cls:
                    break
                else:
                    continue
            except AttributeError or ModuleNotFoundError:
                _logger.debug(f'{self.hparams.activation} fct not found in {m}')
        assert self.loss_fn is not None, f'{self.hparams.loss} could not be found in {_modules_loss}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_optimizer_parameters(self) -> Union[torch.Generator, List[torch.nn.Parameter]]:
        return self.layers.parameters()

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

        try:
            if 'params' in self.hparams.optimizer:
                optimizer = optimizer_cls(self.get_optimizer_parameters(), **self.hparams.optimizer['params'])
            else:
                optimizer = optimizer_cls(self.get_optimizer_parameters())
        except NameError:
            raise NameError(f'Optimizer "{self.hparams.optimizer["type"]}" cannot be found in given '
                            f'sources: "{_modules_optim}"')

        if self.hparams.scheduler['execute']:
            for m in _modules_lr_scheduler:
                try:
                    _, scheduler_cls = dynamic_imp(m, self.hparams.scheduler['type'])
                    scheduler = scheduler_cls(optimizer, **self.hparams.scheduler['params'])
                    if scheduler_cls:
                        break
                    else:
                        continue
                except AttributeError or ModuleNotFoundError:
                    _logger.debug('LR Scheduler of type {} not found in {}'.format(self.hparams.scheduler['type'], m))

            try:
                monitor = 'val_loss'
                return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor}
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
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """
        Validation step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return loss

    def validation_epoch_end(self, outputs: list) -> dict:
        """
        Actions performed at the end of validation epoch (incl. calculating the val_loss)
        """
        val_loss = torch.stack([x for x in outputs]).mean()
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_loss', self.min_val_loss, on_epoch=True, prog_bar=True, logger=False)
        return val_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return loss  # {'test_loss': loss, 'hiddens': hiddens}

    def test_epoch_end(self, outputs: dict) -> dict:
        """
        Actions performed at the end of test epoch (incl. calculating the test_loss)
        """
        test_loss = torch.stack([x for x in outputs]).mean()
        self.log('avg_test_loss', test_loss, logger=True)
        return test_loss

    def hparams_save(self, path: Union[str, Path]) -> None:
        """
        Save hyparams dict to yaml file

        Parameters
        ----------
        path             - path where yaml should be saved
        """
        from pytorch_lightning.core.saving import save_hparams_to_yaml
        save_hparams_to_yaml(path, self.hparams)

    def hparams_update(self, update_dict: dict) -> None:
        """
        Update hyparams dict

        Parameters
        ----------
        update_dict         - dict
        """
        for key, value in update_dict.items():
            self.hparams[key] = value

        if self.get_OptionClass():
            self.check_hparams()
        self.get_functions()

    @staticmethod
    def get_OptionClass() -> dict:
        pass

    @staticmethod
    def yaml_template(key_list: list) -> str:
        pass
