Lightning Models
================

The NN models are the “black box”, the part that learns the patterns in
the data while training is performed. The given toolkit allows an easy
generation of the most common networks via a direct code implementation
or by using the yaml interfaces. In general are all models build by
using the PyTorch Lightning Wrapper that removes the boilerplate of the
PyTorch models without limiting the flexibility. This tutorial will
present the general structure/ possibilities of the implemented models.
However, for more detailed information please have a look at the
documentation of the single models and the `PyTorch Lightning
Docu <https://pytorch-lightning.readthedocs.io/en/0.7.6/lightning-module.html>`__.
The toolbox has pre-implemented a FlexMLP and a more general FlexNN that
can be used for CNNs, RNNs and MLPs. Especially for the FelxNN the usage
of the yaml interface is recommanded in order to have maximal structure.
Own models can be defined using the LightningModelTemplate.py file.

It has to be mentioned that the default dtype of pytorch is changed to
double. This change is necessary in order to export models to C++. As a
consequence, also the input has to be of dtype double.

Initialization methods
----------------------

A Model can be initialized by a direct code implementation or by using
the yaml structure. Thereby, each implemented Model has a yaml template
saved as a staticmethod. In order to employ the yaml file, the utils
functions can be used. The check_argsModel(args_yaml) function is not
mandatory, however the usage is recommanded in order to detect possible
errors and secure that the intended output is provided. In the following
the yaml approache is demonstrated using the LightingFlexMLP:

.. code:: python

    # using the yaml_template
    
    import yaml
    from stfs_pytoolbox.ML_Utils.utils import *
    
    args_yaml = yaml.load(open('path.yaml'), Loader=yaml.FullLoader)
    check_argsModel(args_yaml)
    model = get_model(argsModel)

If a Model is initialized by a direct code implementation, it requires a
Namespace object as input. This object contains all required
hyperparameter used to construct the network, as well as set activation,
loss and optimization function. Three different ways can be identified
in order to create Namespace objects:

1. Create empty Namespace object and add arguments

.. code:: python

    from argparse import Namespace
    from stfs_pytoolbox.ML_Utils import models
    
    hparams = Namespace()
    hparams.key_1 = 'key_value'
    hparams.key_2 = 'int/float/str/dict'
    
    model = models.LightningFlexMLP(hparams)

2. Convert a dict to a Namespace object

.. code:: python

    from argparse import Namespace
    
    hparams_dict = {'key_1': 'key_value', 'key_2': 'int/float/str/dict'}
    hparams = Namespace(**hparams_dict)
    
    model = models.LightningFlexMLP(hparams)

3. Parse arguments using Namespace parser

.. code:: python

    import argparse
    
    hparams_parser = argparse.ArgumentParser()
    hparams_parser.add_argument('--key_1', type=str, default='key_value')
    hparams_parser.add_argument('--key_2', type=str, default='int/float/str/dict')
    hparams = hparams_parser.parse_args()
    
    model = models.LightningFlexMLP(hparams)

Model Hyperparameters
---------------------

Two classes of hyperparameters can be identified: model and algorithm
hyperparameters. The model hyperparameters are used in the model
construction task and thus are not inferred while fitting the network.
However, these parameters influence the learning capability and have to
be adjusted in case the model complexity is chosen differently compared
to the complexity of the addressed problem. In the toolbox theses model
hyperparameters are unique for each model and do not have default
values. As an example the model hyperparameters of the LightningFlexMLP
are:

+-------------+-------+------------------------------------------------+
| key         | dtype | description                                    |
+=============+=======+================================================+
| n_inp:      | int   | Input dimension (required)                     |
+-------------+-------+------------------------------------------------+
| n_out:      | int   | Output dimension (required)                    |
+-------------+-------+------------------------------------------------+
| hidden_laye | list  | List of hidden layers with number of hidden    |
| r:          |       | neurons as layer entry (required)              |
+-------------+-------+------------------------------------------------+

Algorithm hyperparameters, in theory, do not influence the model
performance, instead, they impact the speed and quality of the learning
process. In practice, however, algorithm hyperparameters do influence
the capability and they have to be optimized. Algorithm hyperparameters
are similiar in each model and normally have default values. The toolbox
algorithm hyperparameters are:

+-------------+-------+------------------------------------------------+
| key         | dtype | description                                    |
+=============+=======+================================================+
| activation: | str   | activation fkt that is included in torch.nn    |
|             |       | (default: ReLU)                                |
+-------------+-------+------------------------------------------------+
| loss:       | str   | loss fkt that is included in torch.nn          |
|             |       | (default: MSELoss)                             |
+-------------+-------+------------------------------------------------+
| optimizer:  | dict  | dict including optimizer fkt type and possible |
|             |       | parameters, optimizer has to be included in    |
|             |       | torch.optim (default: {‘type’: Adam, ‘params’: |
|             |       | {‘lr’: 1e-3}})                                 |
+-------------+-------+------------------------------------------------+
| scheduler:  | dict  | dict including execute flag, scheduler fkt     |
|             |       | type and possible parameters, scheduler has to |
|             |       | be included in torch.optim.lr_scheduler        |
|             |       | (default: {‘execute’: False})                  |
+-------------+-------+------------------------------------------------+
| num_workers | int   | number of workers in DataLoaders (default: 10) |
| :           |       |                                                |
+-------------+-------+------------------------------------------------+
| batch:      | int   | batch size of DataLoaders (default: 64)        |
+-------------+-------+------------------------------------------------+
| output_acti | str   | torch.nn activation fkt at the end of the last |
| vation:     |       | layer (default: None)                          |
+-------------+-------+------------------------------------------------+

Each model has to functions in order to save hyperparameter to a yaml
file and update hyperparameters. In the following is a short code
example shown that employs the LightningFlexMLP:

.. code:: python

    from stfs_pytoolbox.ML_Utils import models
    
    model = models.LightningFlexMLP(hparams)
    
    # update hparams by dict or Namespace
    update_hparam = {'loss': RelativeMSELoss, 'optimizer': {'type': 'SGD', 'params': {'lr': 1e-3}}}
    model.hparams_update(update_hparam)
    
    # save hparams to yaml file
    model.hparams_save('some_path.yaml')

Evaluation metrics
------------------

It is possible to use different metrices to evaluate the training of the
network. However, the metrices implemented in pytorch lightning are not
available with version 0.7.6 so that the source code has to be copied
into the metrics directory of the ML_Utils toolbox. As an example this
has been made with the “Explained Variance” metric which is included in
the LightningFlexMLP network. As a consequence of using this metric the
training, validation and test setps/ epoch_end functions have to be
adjusted.
