#######################################################################################################################
# Template to construct Neural Network Models using PyTorch Lightning
#######################################################################################################################

# import packages
import torch
import yaml
import pytorch_lightning as pl
from argparse import Namespace


# flexible MLP class
class LightningModelTemplate(pl.LightningModule):
    """
    Create Model as PyTorch LightningModule
    """

    def __init__(self, hparams):
        """
        Initializes a model with hyperparameters (all parameters included in hparams Namespace are saved automatically
        in the model checkpoint!)
        """
        super().__init__()

        self.hparams = hparams
        self.check_hparams()
        self.get_defaults()
        self.min_val_loss = None

        # code to construct the model
        # store layers in self.layers or change this flag in optimization function
        # store output layer in self.output or change this flag in optimization

    def check_hparams(self) -> None:
        # check hparams important to construct and train the model
        # some tests, which are usually performed, are included

        # check functions
        pass

    def get_defaults(self) -> None:
        # defined default values for hparams
        pass

    def loss_fn(self, y, y_hat):
        """
        compute loss

        Parameters
        ----------
        y           - target tensor of network
        y_hat       - tensor output of network

        Returns
        -------
        loss        - float
        """
        loss_fn = getattr(torch.nn.functional, self.hparams.loss, 'mse_loss')
        return loss_fn(y_hat, y)

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
        for layer in self.layers:
            activation_fn = getattr(torch.nn.functional, self.hparams.activation)
            x = activation_fn(layer(x))

        if self.hparams.output_relu:
            x = torch.nn.functional.relu(self.output(x))
        else:
            x = self.output(x)

        return x

    def configure_optimizers(self):
        """
        optimizer and lr scheduler

        Returns
        -------
        optimizer       - PyTorch Optimizer function
        scheduler       - PyTorch Scheduler function
        """
        params = list(self.layers.parameters()) + list(self.output.parameters())
        if 'params' in self.hparams.optimizer:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(params, **self.hparams.optimizer['params'])
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer['type'])(params)

        if self.hparams.scheduler['execute']:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler['type'], 'ReduceLROnPlateau') \
                (optimizer, **self.hparams.scheduler['params'])
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        log = {'train_loss': loss}
        results = {'loss': loss, 'log': log}
        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y, y_hat)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.current_epoch == 0: self.min_val_loss = val_loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        log = {'avg_val_loss': val_loss}
        pbar = {'val_loss': val_loss, 'min_val_loss': self.min_val_loss}
        results = {'log': log, 'val_loss': val_loss, 'progress_bar': pbar}
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
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

    @staticmethod
    def yaml_template(key_list):
        template = {'Model': {'type': 'LightningModelTemplate',
                              'load_model': {"INFO": 'Mutually exclusive with create_model', 'path': 'name.ckpt'},
                              'create_model': {"INFO": 'Mutually exclusive with load_model',
                                               'some_construction_parameters': 'corresponding datatypes',
                                               'output_relu': 'bool (default: False)', 'activation': 'relu'},
                              'params': {'loss': 'mse_loss', 'optimizer': {'type': 'Adam', 'params': {'lr': 1.e-3}},
                                         'scheduler': {'execute': ' bool (default: False)', 'type': 'name',
                                                       'params': {'param_1': 'int', 'param_2': 'int'}},
                                         'num_workers': 'int', 'batch': 'int'}}}

        for i, key in enumerate(key_list):
            template = template.get(key)

        return yaml.dump(template, sort_keys=False)
