Working Example: Mnist
======================

In order to demonstarte how the Toolbox can be employed for
Two-Dimensional (e.g. Images) input data in a classification task, this
example uses the Mnist Dataset. The set includes 60000 training and
10000 test images with a size of 28x28 pixels. In total 10 different
clases can be identified. As mentioned, there are different ways to
implement the model. Here both strategies (yaml and code interface) are
demonstrated.

The example is structured as follows: 1. Load the Mnist dataset 2.
Generate Model with yaml file 3. Generate Model by code implementation
4. Train model

Load Mnist Dataset
------------------

The Mnist Data set is included in the torchvision datasets. In general,
it is possible to load the set and pass it nearly into the
torch.utils.data.DataLoader method. However, with the toolbox this is a
little bit tricky since the models dtype is set to double. This default
change is necessary in order to export the model to C++. As a
consequence the model expects a double input so that the dataset tensor
has to be changed manually to double and then the channels have to be
added by reshaping the images.

.. code:: python

    import torch
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform) 
    x_train = mnist_train.data.double()
    x_train = x_train.reshape((60000, 1, 28, 28))
    y_train = mnist_train.targets.double()
    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False, num_workers=10)
    
    mnist_val = datasets.MNIST('./data', train=False, download=True, transform=transform)
    x_val = mnist_val.data.double()
    x_val = x_val.reshape((10000, 1, 28, 28))
    y_val = mnist_val.targets.double()
    val = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False, num_workers=10)

Create model with yaml interface
--------------------------------

Using the yaml interface is considered to be more structured and less
likely to fail. In case of this example, only the model is creating
using a yaml file, the trainer as well as the data are directly
implemented. In order to construct the yaml, you don’t have to start
from scratch since each model has a pre-implemented yaml template. To
obtain the yaml file, we just call the staticmethod yml_template, give
it an empty list and then copy the output into a yaml file:

.. code:: python

    from stfs_pytoolbox.ML_Utils.models import LightningFlexNN
    
    print(LightningFlexNN.yaml_template([]))


.. parsed-literal::

    Model:
      type: LightningFlexNN
      '###INFO###': load_model and create_model are mutually exclusive
      load_model:
        path: name.ckpt
      create_model:
        width: int
        height: int
        depth: int
        layers:
        - type: torch.nn module
          params:
            module_param_1: value
            module_param_2: value
        - type: e. g. Conv2d
          params:
            kernel_size: 3
            channels: 20
        - type: e. g. MaxPool2d
          params:
            kernel_size: 2
        MLP_layer:
          n_out: int
          hidden_layer:
          - int
          - int
          - '...'
        output_activation: 'str (default: None)'
        activation: 'str (default: ReLU)'
      params:
        loss: str (default:MSELoss)
        optimizer:
          type: 'str (default: Adam)'
          params:
            lr: 'float (default: 1.e-3'
        scheduler:
          execute: ' bool (default: False)'
          type: name
          params:
            cooldown: int
            patience: int
            min_lr: float
        num_workers: 'int (default: 10)'
        batch: 'int (default: 64)'
    


Model:
  type: LightningFlexNN
  create_model:
    width: 28
    height: 28
    depth: 1
    layers:
    - type: Conv2d
      params:
        kernel_size: 3
        channels: 16
    - type: MaxPool2d
      params:
        kernel_size: 2
    MLP_layer:
      n_out: 10
      hidden_layer:
      - 64
    output_activation: LogSigmoid
  params:
    loss: CrossEntropyLoss

After adjusting the yaml file for the used case, the model is created
using the utils function “get_model”. It is further recommended to use
check_argsmodel in order to detect possible mistakes made while changing
the yaml file.

.. code:: python

    from stfs_pytoolbox.ML_Utils.utils import get_model, check_argsModel
    import yaml
    
    argsYaml = yaml.load('path.yaml', Loader=yaml.FullLoader)
    check_argsModel(argsYaml['Model'])
    model = get_model(argsYaml['Model'])

Create Model by direct implementation
-------------------------------------

Different ways to construct the Namespace object needed to construct a
model are given in `Model Documentation <>`__. Here the Namespace is
converted out of a dict. In order to quickly generated the dict, it can
be copied out of the yml template function of the used model and then
adjusted. It is possible to pass the dict again to the get_model
function. Here, however, you can see the steps that are basically
performed. Thereby, the model is firstly created and the updated by the
hyperparameters defined in “params”.

.. code:: python

    from stfs_pytoolbox.ML_Utils.models import LightningFlexNN
    import argparse
    
    model_dict = {'create_model': {'width': 28, 'height': 28, 'depth': 1,
                                   'layers': [{'type': 'Conv2d', 'params': {'kernel_size': 3, 'channels': 16, 'stride': 1}},
                                              {'type': 'MaxPool2d', 'params': {'kernel_size': 2}}],
                                   'MLP_layer': {'n_out': 10, 'hidden_layer': [64]}},
                  'params': {'loss': 'CrossEntropyLoss'}}
    
    model = LightningFlexNN(argparse.Namespace(**model_dict['create_model']))
    model.hparams_update(model_dict['params'])

Train model
-----------

Training is performed using the Lighting Trainer class. Since in this
example we only want to control that the model is working correctly, the
fast_dev_run flag is set to True.

.. code:: python

    import pytorch_lightning as pl
    
    trainer = pl.Trainer(fast_dev_run=True, logger=False)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
