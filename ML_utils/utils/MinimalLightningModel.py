import torch
from argparse import Namespace

import pytorch_lightning as pl


class MinimalLightningModule(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        self.hparams = hparams
        self.activation_fn = torch.nn.ReLU()
        # Construct MLP with a variable number of hidden layers
        self.layers = []
        self.construct_mlp(self.hparams.n_inp, self.hparams.hidden_layer, self.hparams.n_out)
        self.layers = torch.nn.Sequential(*self.layers)

    def construct_mlp(self, in_dim, hidden_layer, out_dim) -> None:
        # Construct all MLP layers
        self.layers.append(torch.nn.Linear(in_dim, hidden_layer[0]))
        self.layers.append(self.activation_fn)

        layer_sizes = zip(hidden_layer[:-1], hidden_layer[1:])

        for h1, h2 in layer_sizes:
            self.layers.append(torch.nn.Linear(h1, h2))
            self.layers.append(self.activation_fn)

        self.layers.append(torch.nn.Linear(hidden_layer[-1], out_dim))

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        return {'loss': loss}
