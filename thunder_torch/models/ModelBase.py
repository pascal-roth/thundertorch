#######################################################################################################################
# Base for all models in the Toolbox, includes basic functions
#######################################################################################################################

# import packages
import torch
import importlib
import pytorch_lightning as pl
from argparse import Namespace
from typing import List, Union, Tuple
from pathlib import Path
from collections.abc import Callable

from thunder_torch import _logger
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_optim, _modules_lr_scheduler
from thunder_torch.utils.general import dynamic_imp


# flexible MLP class
class LightningModelBase(pl.LightningModule):
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
        self.min_val_loss:  float
        self.final_channel: int

        self.layers:        List = []
        self.height:        int
        self.width:         int
        self.depth:         int
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,)

    def construct_nn2d(self, layer_list: dict) -> None:
        """
        Functionality to build any kind of torch.nn 2d layer (convolutional, pooling, padding, normalization, recurrent,
        dropout, linear ...)

        Parameters
        ----------
        layer_list          - list of layers with the corresponding parameters
        """
        # Construct all conv layers
        for layer_dict in layer_list:
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

    def construct_nn3d(self, layer_list: dict) -> None:
        """
        Functionality to build any kind of torch.nn 3d layer (convolutional, pooling, padding, normalization, recurrent,
        dropout, linear ...)

        Parameters
        ----------
        layer_list          - list of layers with the corresponding parameters
        """
        # Construct all conv layers
        for layer_dict in layer_list:
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
                    self.depth = int((self.depth + 2 * self.layers[-1].padding[2]) /
                                     self.layers[-1].stride[2]) - (self.layers[-1].kernel_size[2] - 1)
                else:
                    self.height = int((self.height + 2 * self.layers[-1].padding) /
                                      self.layers[-1].stride) - (self.height % self.layers[-1].kernel_size)
                    self.width = int((self.width + 2 * self.layers[-1].padding) /
                                     self.layers[-1].stride) - (self.width % self.layers[-1].kernel_size)
                    self.depth = int((self.depth + 2 * self.layers[-1].padding) /
                                     self.layers[-1].stride) - (self.width % self.layers[-1].kernel_size)

            if isinstance(self.layers[-1], self.layer_activation):
                self.layers.append(self.activation_fn)

    def construct_mlp(self, in_dim: int, hidden_layer: List[int], out_dim: int) -> None:
        """
        Quick functionality to built a MLP network

        Parameters
        ----------
        in_dim              - input dimensions
        hidden_layer        - dimensions of the hidden layers
        out_dim             - output dimensions
        """
        # TODO: think about adding a bias flag in case a linear function should be learned
        # Construct all MLP layers
        self.layers.append(torch.nn.Linear(in_dim, hidden_layer[0]))
        self.layers.append(self.activation_fn)

        layer_sizes = zip(hidden_layer[:-1], hidden_layer[1:])

        for h1, h2 in layer_sizes:
            self.layers.append(torch.nn.Linear(h1, h2))
            self.layers.append(self.activation_fn)

        self.layers.append(torch.nn.Linear(hidden_layer[-1], out_dim))

    def check_hparams(self) -> None:
        options = self.get_OptionClass()
        options["hparams"].add_key('model_type', dtype=str)
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
        if not hasattr(self.hparams, 'model_type'):
            class_name = str(self.__class__)
            class_name_split = class_name.split("'")[1]
            model_type = class_name_split.split(".")[-1]
            self.hparams.model_type = model_type

    def get_functions(self) -> None:
        """
        get activation and loss function from a source defined in _modules_activation or _modules_loss
        """
        for m in _modules_activation:
            try:
                _, activation_cls = dynamic_imp(m, self.hparams.activation)
                self.activation_fn = activation_cls()
                # self.activation_fn = getattr(importlib.import_module(m), self.hparams.activation)()
                _logger.debug(f'{self.hparams.activation} fct found in {m}')
                break
            except AttributeError or ModuleNotFoundError:
                _logger.debug(f'{self.hparams.activation} fct not found in {m}')
        assert self.activation_fn is not None, f'{self.hparams.activation} could not be found in {_modules_activation}'

        for m in _modules_loss:
            try:
                _, loss_cls = dynamic_imp(m, self.hparams.loss)
                self.loss_fn = loss_cls()
                # self.loss_fn = getattr(importlib.import_module(m), self.hparams.loss)()
                _logger.debug(f'{self.hparams.activation} fct found in {m}')
                break
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

        if 'params' in self.hparams.optimizer:
            optimizer = optimizer_cls(self.layers.parameters(), **self.hparams.optimizer['params'])
        else:
            optimizer = optimizer_cls(self.layers.parameters())

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
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Training step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        try:
            loss = self.loss_fn(y_hat, y)
        except RuntimeError:  # TODO: makes target to int, really useful ?
            loss = self.loss_fn(y_hat, y.long())
        log = {'train_loss': loss}
        hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        results = {'loss': loss, 'log': log, 'hiddens': hiddens}
        return results

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Validation step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        try:
            loss = self.loss_fn(y_hat, y)
        except RuntimeError:
            loss = self.loss_fn(y_hat, y.long())
        hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return {'val_loss': loss, 'hiddens': hiddens}

    def validation_epoch_end(self, outputs: dict) -> dict:
        """
        Actions performed at the end of validation epoch (incl. calculating the val_loss)
        """
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        log = {'avg_val_loss': val_loss}
        pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}
        results = {'log': log, 'val_loss': val_loss, 'progress_bar': pbar}
        return results

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        try:
            loss = self.loss_fn(y_hat, y)
        except RuntimeError:
            loss = self.loss_fn(y_hat, y.long())
        hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return {'test_loss': loss, 'hiddens': hiddens}

    def test_epoch_end(self, outputs: dict) -> dict:
        """
        Actions performed at the end of test epoch (incl. calculating the test_loss)
        """
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'avg_test_loss': test_loss}
        results = {'log': log, 'test_loss': test_loss}
        return results

    def hparams_save(self, path: Union[str, Path]) -> None:
        """
        Save hyparams dict to yaml file

        Parameters
        ----------
        path             - path where yaml should be saved
        """
        from pytorch_lightning.core.saving import save_hparams_to_yaml
        save_hparams_to_yaml(path, self.hparams)

    def hparams_update(self, update_dict: Union[dict, Namespace]) -> None:
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
        if self.get_OptionClass():
            self.check_hparams()
        self.get_functions()

    @staticmethod
    def get_OptionClass() -> dict:
        pass

    @staticmethod
    def yaml_template() -> str:
        pass
