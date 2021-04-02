Working Example: Code implementation
====================================

The general usage of the ML Toolbox should be demonstrated here.
Therefore, the following elements are included in this Example:

-  Configuration of the training (optional)
-  Creation of a DataLoader
-  Creation of a LightningModel
-  Training and testing of the Model
-  Model Loading and training continuation

Thereby, the TabularLoader and the LightningFlexMLP are used.

Configuration
-------------

In the configuration individual modules can be added and the
reproducibility option of the training activated. For a detailed
explanation of the individual modules can be found
`here <../getting_started/Individualized_modules.html>`__. If the
reproducibility option it is selected the random seed of PyTorch and
Numpy a set to fix value. The configuration is executed as follows:

.. code:: python

    from stfs_pytoolbox.ML_Utils.utils import train_config
    
    argsConfig = {'source_files': module_path, 'reproducibility': True}
    train_config(argsConfig)

DataLoader
----------

In the first step, a random dataframe is created and used as an input.
The data is split into training, validation, and test datasets. In case
of the validation split, the “random” method is used and 20% of the
dataset is separated. The test data is separated using the “percentage”
method that is, due to the randomly created input, equal to the random
method.

.. code:: python

    import numpy as np
    import pandas as pd
    
    from stfs_pytoolbox.ML_Utils import loader
    
    example_df = pd.DataFrame(np.random.rand(10000, 5))
    example_df.columns = ['T_0', 'P_0', 'PV', 'yCO2', 'wH2O']
    
    features = ['T_0', 'P_0', 'PV']
    labels = ['yCO2', 'wH2O']
    
    dataLoader = loader.TabularLoader(example_df, features, labels, val_split={'method': 'random', 'params': 0.2}, test_split={'method': 'percentage', 'params': {'T_0': 0.1}})

LightningModule
---------------

Secondly, the NN is created. The possible hyperparameters for the
differents model are included in their description. In the case of the
FlexMLP those parameters are:

+--------------+------+------------------------------------------------+
| key          | dtyp | description                                    |
|              | e    |                                                |
+==============+======+================================================+
| n_inp:       | int  | Input dimension (required)                     |
+--------------+------+------------------------------------------------+
| n_out:       | int  | Output dimension (required)                    |
+--------------+------+------------------------------------------------+
| hidden_layer | list | List of hidden layers with number of hidden    |
| :            |      | neurons as layer entry (required)              |
+--------------+------+------------------------------------------------+
| activation:  | str  | activation fkt that is included in torch.nn    |
|              |      | (default: ReLu)                                |
+--------------+------+------------------------------------------------+
| loss:        | str  | loss fkt that is included in torch.nn          |
|              |      | (default: MSELoss)                             |
+--------------+------+------------------------------------------------+
| optimizer:   | dict | dict including optimizer fkt type and possible |
|              |      | parameters, optimizer has to be included in    |
|              |      | torch.optim (default: {‘type’: Adam, ‘params’: |
|              |      | {‘lr’: 1e-3}})                                 |
+--------------+------+------------------------------------------------+
| scheduler:   | dict | dict including execute flag, scheduler fkt     |
|              |      | type and possible parameters, scheduler has to |
|              |      | be included in torch.optim.lr_scheduler        |
|              |      | (default: {‘execute’: False})                  |
+--------------+------+------------------------------------------------+
| num_workers: | int  | number of workers in DataLoaders (default: 10) |
+--------------+------+------------------------------------------------+
| batch:       | int  | batch size of DataLoaders (default: 64)        |
+--------------+------+------------------------------------------------+
| output_activ | str  | torch.nn activation fkt at the end of the last |
| ation:       |      | layer (default: None)                          |
+--------------+------+------------------------------------------------+

The required hyperparameters are the input and output dimension, as well
as the number of hidden layers with the corresponding number of hidden
units for each layer. Furthermore, some other hyperparameters are
changed in this example. In particular, the loss and optimizer function
are adjusted. As discussed in the Models Tutorial, different ways exist
to create the Namespace object used as input for the network.

.. code:: python

    import argparse
    
    from stfs_pytoolbox.ML_Utils import models
    
    hparams = argparse.Namespace()
    hparams.n_inp = 3
    hparams.n_out = 2
    hparams.hidden_layer = [32, 64]
    hparams.loss = 'RelativeMSELoss'
    hparams.optimizer = {'type': 'SGD', 'params': {'lr': 1e-3}}
    
    model = models.LightningFlexMLP(hparams)

In order to allow us to recreate the DataLoader, the necessary
information have to be included in the Model checkpoint. Since the
DataLoader, in its initialization process, aims to load the used data
file, the randomly created DataFrame is saved here as .csv file and the
location is passed in the DataLoader params Namespace “lparams” as
“data_path”. In case the DataLoader has been generated directly from a
file, the data_path is saved automatically. To include the lparams
Namespace object, the hparams_update function is used. However, the
lparams Namespace has to be included in its own dict. This is necessary
since the hparams object has a key controller that does not include each
loader key but instead just the lparams key.

.. code:: python

    data_path = 'example_samples.csv'
    
    example_df.to_csv(data_path)  # save DataFrame
    dataLoader.lparams.data_path = data_path  # include data_path in dataLoader lparams
    
    model.hparams_update({'lparams': dataLoader.lparams})

Training and Testing
--------------------

Training and testing are performed using the pl.Trainer. In the case of
this example, we just want to control whether our network is fully
functional and that our network training can be continued by saving the
model checkpoint. Therefore the checkpointing callback is active and
max_epochs equals 1. The callback has its own flag and has been modified
so that the epoch is only added to the path if multiple models should be
saved.

.. code:: python

    import pytorch_lightning as pl
    
    from stfs_pytoolbox.ML_Utils import callbacks
    
    checkpointing = callbacks.Checkpointing(filepath='checkpoints/model_example')
    
    trainer = pl.Trainer(max_epochs=1, logger=False, checkpoint_callback=checkpointing)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())

Model Loading and Training Continuation
---------------------------------------

Let’s say that a two-stage training is intended whereas in the first
stage the “RelativeMSELoss” and in the second stage, the normal
“mse_loss” is used. In this case, the model has to be loaded and the
dataLoader has to be regenerated (if not started in the same script).
Thereby, model and dataLoader types have to be known. To further train
the pre-trained weights, it is crucial that also the trainer flag
“resume_from_checkpoint” is set. Here the procedure will be demonstrated
shortly.

.. code:: python

    model2 = models.LightningFlexMLP.load_from_checkpoint('checkpoints/model_example.ckpt')
    dataLoader2 = loader.TabularLoader.read_from_checkpoint('checkpoints/model_example.ckpt')
    
    checkpointing2 = callbacks.Checkpointing(filepath='checkpoints/model_example_retrained')
    model.hparams_update({'loss': 'MSELoss'})
    
    trainer2 = pl.Trainer(max_epochs=2, logger=False, checkpoint_callback=checkpointing, resume_from_checkpoint='checkpoints/model_example.ckpt')
    trainer2.fit(model2, train_dataloader=dataLoader2.train_dataloader(), val_dataloaders=dataLoader2.val_dataloader())
    trainer2.test(model2, test_dataloaders=dataLoader2.test_dataloader())
