#######################################################################################################################
# Base for all models in the Toolbox, includes basic functions
#######################################################################################################################

# import packages
import torch
import yaml
import pytorch_lightning as pl
from argparse import Namespace

from stfs_pytoolbox.ML_Utils.models import _losses
from stfs_pytoolbox.ML_Utils.utils.option_class import OptionClass
from stfs_pytoolbox.ML_Utils import metrics


# flexible MLP class
class LightningModelBase(pl.LightningModule):
    """
    Model Base of the Toolbox, includes repeading functions
    """

    def __init__(self):
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

        self.layers = []
        self.height = None
        self.width = None
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,)

    def construct_nn(self, layer_list) -> None:
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

    def construct_mlp(self, in_dim, hidden_layer, out_dim) -> None:
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
            self.loss_fn = getattr(_losses, self.hparams.loss)()

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
        log = {'train_loss': loss}
        results = {'loss': loss, 'log': log}
        return results

    def validation_step(self, batch, batch_idx) -> dict:
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y.long())
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs) -> dict:
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.current_epoch == 0: self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        log = {'avg_val_loss': val_loss}
        pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}
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
        log = {'avg_test_loss': test_loss}
        results = {'log': log, 'test_loss': test_loss}
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
