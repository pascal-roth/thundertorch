import torch
import pytorch_lightning as pl


class flexCNN_pl(pl.LightningModule):
    def __init__(self, labels, features, n_hidden_neurons=[32, 32], activation_fn=torch.nn.functional.relu,
                  lr=1e-3, loss_fn=torch.nn.functional.mse_loss):

        super().__init__()

        # check for features and labels
        if not isinstance(features, list):
            features = [features]
        n_inp = len(features)
        for i in range(n_inp):
            assert isinstance(features[i], str), "init FlexMLP_pl: Given features is not a list of strings!"

        if not isinstance(labels, list):
            features = [labels]
        n_out = len(labels)
        for i in range(n_out):
            assert isinstance(labels[i], str), "init FlexMLP_pl: Given labels is not a list of strings!"

        self.features = features
        self.labels = labels

        self.n_inp = n_inp
        self.n_out = n_out

        self.lr = lr
        self.loss = loss_fn
        self.activation_fn = activation_fn

        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_inp, n_hidden_neurons[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(n_hidden_neurons[:-1], n_hidden_neurons[1:])
        self.layers.extend([torch.nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = torch.nn.Linear(n_hidden_neurons[-1], n_out)

    def forward(self, x):
        for each in self.layers:
            x = self.activation_fn(each(x))
        x = self.output(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, patience=10, min_lr=1e-8)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss}
        return {'log': log, 'val_loss': val_loss}

    # def save(self, path):
    #     createFlexMLPCheckpoint(self.model, path, self.features, self.labels, self.scalers)

