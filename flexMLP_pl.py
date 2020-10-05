#######################################################################################################################
# flexible MLP model constructed using PyTorch and the PyTorch Lightning Wrapper Tool
#######################################################################################################################
# Problem with Tensorboard (officially available only with pytorch 1.3
#    - add function hparams to torch/utils/tensorboard/summary.py
#    - remove version check in pytorch_lightning/logggers/tensorboard.py
# idea define own logger where all code is copied and just those changes implemented
# access saved data by script and perform plotting: https://www.tensorflow.org/tensorboard/dataframe_api

# import packages
import torch
import pytorch_lightning as pl
from sklearn import preprocessing
from losses import RelativeMSELoss

# flexible MLP class
class flexMLP_pl(pl.LightningModule):
    """
    Create flexMLP as PyTorch LightningModule
    """
    def __init__(self, hparams, TabularLoader):
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters (features, labels, lr, activation fn, ...)
        TabularLoader   - TabularLoader object including training, validation and test samples as pd.DataFrames
        """
        super().__init__()

        self.hparams = hparams
        self.TabularLoader = TabularLoader

        # Built model
        self.hparams.n_inp = len(self.hparams.features)
        self.hparams.n_out = len(self.hparams.labels)

        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.hparams.n_inp, self.hparams.hidden_layer[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(self.hparams.hidden_layer[:-1], self.hparams.hidden_layer[1:])
        self.layers.extend([torch.nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = torch.nn.Linear(self.hparams.hidden_layer[-1], self.hparams.n_out)

    def prepare_data(self):
        """
        seperate into samples and targets, normalize both
        """
        if self.TabularLoader.samples_train is not None:
            self.x_train = self.TabularLoader.samples_train[self.hparams.features]
            self.y_train = self.TabularLoader.samples_train[self.hparams.labels]
            self.TabularLoader.samples_train = None

            if self.hparams.x_scaler is None:
                x_min_max_scaler = preprocessing.MinMaxScaler()
                self.hparams.x_scaler = x_min_max_scaler.fit(self.x_train)  # TODO: indem ich hier die Trainingsdaten nehmen, gibt es die Gefahr, dass ich einen Wert auch größer als 1 im test oder val set habe ...
            if self.hparams.y_scaler is None:
                y_min_max_scaler = preprocessing.MinMaxScaler()
                self.hparams.y_scaler = y_min_max_scaler.fit(self.y_train)

            self.x_train = self.hparams.x_scaler.transform(self.x_train)
            self.y_train = self.hparams.y_scaler.transform(self.y_train)
        else:
            raise KeyError('No training data included!')

        if self.TabularLoader.samples_val is not None:
            self.x_val = self.TabularLoader.samples_val[self.hparams.features]
            self.y_val = self.TabularLoader.samples_val[self.hparams.labels]
            self.x_val = self.hparams.x_scaler.transform(self.x_val)
            self.y_val = self.hparams.y_scaler.transform(self.y_val)
            self.TabularLoader.samples_val = None
        else:
            raise KeyError('No validation data included!')

        if self.TabularLoader.samples_test is not None:
            self.x_test = self.TabularLoader.samples_test[self.hparams.features]
            self.y_test = self.TabularLoader.samples_test[self.hparams.labels]
            self.x_test = self.hparams.x_scaler.transform(self.x_test)
            self.y_test = self.hparams.y_scaler.transform(self.y_test)
            self.TabularLoader.samples_test = None
        else:
            raise KeyError('No test data included!')

    def activation_fn(self, x, each):
        """
        compute layer output according to the activation function

        Parameters
        ----------
        x           - input tensor of the pytorch.nn.Linear layer
        each        - pytorch.nn.Linear layer

        Returns
        -------
        x           - output tensor of the pytorch.nn.Linear layer
        """
        # select activation function
        assert isinstance(self.hparams.activation, str), 'Activation function name has to be of type str'

        if self.hparams.activation == 'relu':  # TODO: wäre schöner, wenn man hier den str einfach nur an torch.nn.functional hängen könnte
            x = torch.nn.functional.relu(each(x))
        elif self.hparams.activation == 'tanh':
            x = torch.nn.functional.tanh(each(x))
        elif self.hparams.activation == 'softplus':
            x = torch.nn.functional.softplus(each(x))
        else:
            raise NotImplementedError('Activation function {} not available, has to add manually'.format(self.hparams.activation))

        return x

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
        # select loss function
        assert isinstance(self.hparams.loss, str), 'Loss function name has to be of type str'

        if self.hparams.loss == 'MSE':
            loss = torch.nn.functional.mse_loss(y_hat, y)
        else:
            raise NotImplementedError('Selected loss function not implemented yet!')

        return loss

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
            x = self.activation_fn(x, layer)

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
        assert isinstance(self.hparams.optimizer, str), 'Optimizer function name has to be of type str'

        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('{} optimizer not implemented yet!'.format(self.hparams.optimizer))

        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, patience=10, min_lr=1e-8)
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
        log = {'avg_val_loss': val_loss}
        results = {'log': log, 'val_loss': val_loss}
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        log = {'test_loss': loss}
        return {'loss': loss, 'log': log}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'avg_test_loss': test_loss}
        results = {'log': log, 'test_loss': test_loss}
        return results

    def train_dataloader(self):
        x_samples = torch.tensor(self.x_train).float()
        y_samples = torch.tensor(self.y_train).float()
        tensor = torch.utils.data.TensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.hparams.batch, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        x_samples = torch.tensor(self.x_val).float()
        y_samples = torch.tensor(self.y_val).float()
        tensor = torch.utils.data.TensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.hparams.batch, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        x_samples = torch.tensor(self.x_test).float()
        y_samples = torch.tensor(self.y_test).float()
        tensor = torch.utils.data.TensorDataset(x_samples, y_samples)
        return torch.utils.data.DataLoader(tensor, batch_size=self.hparams.batch, num_workers=self.hparams.num_workers)

    # def add_model_specific_args(parent_parser):
    #     parser = argparse.ArgumentParser(parents=[parent_parser])
    #     parser.add_argument('--features', type=list, default=['pode', 'Z', 'H', 'PV'])
    #     parser.add_argument('--labels', type=list, default=['T'])
    #     parser.add_argument('--n_hidden_neurons', nargs='+', type=int, default=[64, 64, 64])
    #     return parser