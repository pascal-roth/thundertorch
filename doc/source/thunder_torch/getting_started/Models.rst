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
can be used for CNNs, RNNs, and MLPs. Especially for the FelxNN, the
usage of the yaml interface is recommended in order to have maximal
possible structure. Own models can be defined using the
LightningModelTemplate.py file.

It has to be mentioned that the default dtype of pytorch is changed to
double. This change is necessary in order to export models to C++. As a
consequence, also the input has to be of dtype double.

Initialization methods
----------------------

A Model can be initialized by a direct code implementation or by using
the yaml structure. Thereby, each implemented Model has a yaml template
saved as a staticmethod. In order to employ the yaml file, the utils
functions can be used. The check_argsModel(args_yaml) function is not
mandatory, however, the usage is recommended in order to detect possible
errors and secure that the intended output is provided. In the following
the yaml approach is demonstrated using the LightingFlexMLP:

.. code:: python

    # using the yaml_template
    
    import yaml
    from thunder_torch.utils import *
    
    args_yaml = parse_yaml('path.yaml')
    check_argsModel(args_yaml['model'])
    model = get_model(argsModel['model'])

If a Model is initialized by a direct code implementation, it requires a
Namespace object as input. This object contains all required
hyperparameter used to construct the network, as well as set activation,
loss, and optimization function. Three different ways can be identified
in order to create Namespace objects:

1. Create empty Namespace object and add arguments

.. code:: python

    from argparse import Namespace
    from thunder import models
    
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
to the complexity of the addressed problem. In the toolbox, theses model
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
are similar in each model and normally have default values. The toolbox
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

    from thunder import models
    
    model = models.LightningFlexMLP(hparams)
    
    # update hparams by dict or Namespace
    update_hparam = {'loss': RelativeMSELoss, 'optimizer': {'type': 'SGD', 'params': {'lr': 1e-3}}}
    model.hparams_update(update_hparam)
    
    # save hparams to yaml file
    model.hparams_save('some_path.yaml')

LightningModelBase and Individual Models
----------------------------------------

The Toolbox has an own ModelBase class which contains the repeading
functions like training, validation and test step. This ModelBase Class
furthermore has two functionalities that can construct most of the
network layers which are included in torch.nn so that most models can be
constructed by just using these functions.

However, if the addressed task cannot be solved using the
pre-implemented methods, individual modules can be constructed and then
used instead. A detailed explanation on how to include individual models
in the toolbox can be found `here <./Individualized_modules>`__. It is
important to keep in mind that the functions defined in the
thunder_torch.model._LightningModelTemplate.LightningModelTemplate
have to be included since they will be used in the training procedure.

Model Extensions
----------------

Model functionalities can be extended using different callbacks or
metrics. Detailed explanations can be found here:

-  `Callbacks <./Callbacks.html>`__
-  `Metrics <./Metrics.html>`__
