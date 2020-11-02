import pytest
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets, transforms

from stfs_pytoolbox.ML_Utils.models import LightningFlexNN

def test_LightningFlexNN_integration(tmp_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = datasets.MNIST(tmp_path, train=True, download=True, transform=transform) # TODO: change to tmp_dir
    x_train = mnist_train.data[:1000].double()
    x_train = x_train.reshape((1000, 1, 28, 28))
    y_train = mnist_train.targets[:1000].double()
    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False, num_workers=10)

    mnist_val = datasets.MNIST(tmp_path, train=False, download=True, transform=transform)
    x_val = mnist_val.data[:200].double()
    x_val = x_val.reshape((200, 1, 28, 28))
    y_val = mnist_val.targets[:200].double()
    val = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False, num_workers=10)

    model_dict = {'create_model': {'width': 28, 'height': 28, 'depth': 1,
                                   'layers': [{'type': 'Conv2d', 'params': {'kernel_size': 3, 'channels': 16, 'stride': 1}},
                                              {'type': 'MaxPool2d', 'params': {'kernel_size': 2}}],
                                   'MLP_layer': {'n_out': 10, 'hidden_layer': [64]}},
                  'params': {'loss': 'CrossEntropyLoss'}}

    model = LightningFlexNN(argparse.Namespace(**model_dict['create_model']))
    model.hparams_update(model_dict['params'])

    for layer in model.layers:
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, )):
            torch.nn.init.constant_(layer.weight, val=0.1)
            torch.nn.init.constant_(layer.bias, val=0.1)

    trainer = pl.Trainer(max_epochs=3, logger=False)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # compare training, validation and test loss
    val_losses = trainer.tng_tqdm_dic
    assert np.round(val_losses['val_loss'], decimals=5) == 117.11979, {'Integration test failed'}
    assert val_losses['loss'] == '304.758', {'Integration test failed'}
