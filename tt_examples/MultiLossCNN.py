#######################################################################################################################
# flexible CNN model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################

# import packages
import torch
import yaml
from argparse import Namespace
from typing import List, Tuple

from thunder_torch.models.ModelBase import LightningModelBase
from thunder_torch.utils.option_class import OptionClass
from thunder_torch import _modules_activation, _modules_loss, _modules_lr_scheduler, _modules_optim
import thunder_torch as tt
from thunder_torch.models.ContinueEnCoder import LightningFlexContinueModel


class LightningFlexNNMultiLoss(LightningFlexContinueModel):
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
        super().__init__(hparams)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Training step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss_mse = torch.nn.MSELoss()(y_hat, y)
        loss_rel = tt.models.RelativeMSELoss()(y_hat, y)
        loss = loss_mse + loss_rel
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
        loss_mse = torch.nn.MSELoss()(y_hat, y)
        loss_rel = tt.models.RelativeMSELoss()(y_hat, y)
        loss = loss_mse + loss_rel
        hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return {'val_loss': loss, 'hiddens': hiddens}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Test step of the network
        the "hiddens" dicts saves parameters which can be used in callback
        """
        x, y = batch
        y_hat = self(x)
        loss_mse = torch.nn.MSELoss()(y_hat, y)
        loss_rel = tt.models.RelativeMSELoss()(y_hat, y)
        loss = loss_mse + loss_rel
        hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}
        return {'test_loss': loss, 'hiddens': hiddens}
