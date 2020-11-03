Working Example: Code implementation
====================================

The general usage of the ML Toolbox should be demonstarted here with.
Therefore, following elements are included in this Example:

-  Creation of a DataLoader
-  Creation of a LightningModel
-  Training and testing of the Model
-  Model Loading and training continuation

Thereby, the TabularLoader and the LightningFlexMLP are used.

DataLoader
----------

In the first step, a random dataframe is created and used as an input.
The data is splitted into training, validation and test data set. In
case of the validation split, the “random” method is used and 20% of the
dataset is seperated. The test data is separated using the “percentage”
method that is, due to the randomely created input, equal to the random
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

Secondely, the NN is created. The possible hyperparameters for the
differents model are included in their discription. In case of the
FlexMLP those parameters are:

+------+---+----------------------------------------------------------+
| key  | d | description                                              |
|      | t |                                                          |
|      | y |                                                          |
|      | p |                                                          |
|      | e |                                                          |
+======+===+==========================================================+
| n_in | i | Input dimension (required)                               |
| p:   | n |                                                          |
|      | t |                                                          |
+------+---+----------------------------------------------------------+
| n_ou | i | Output dimension (required)                              |
| t:   | n |                                                          |
|      | t |                                                          |
+------+---+----------------------------------------------------------+
| hidd | l | List of hidden layers with number of hidden neurons as   |
| en_l | i | layer entry (required)                                   |
| ayer | s |                                                          |
| :    | t |                                                          |
+------+---+----------------------------------------------------------+
| acti | s | activation fkt that is included in torch.nn (default:    |
| vati | t | ReLu)                                                    |
| on:  | r |                                                          |
+------+---+----------------------------------------------------------+
| loss | s | loss fkt that is included in torch.nn (default: MSELoss) |
| :    | t |                                                          |
|      | r |                                                          |
+------+---+----------------------------------------------------------+
| opti | d | dict including optimizer fkt type and possible           |
| mize | i | parameters, optimizer has to be included in torch.optim  |
| r:   | c | (default: {‘type’: Adam, ‘params’: {‘lr’: 1e-3}})        |
|      | t |                                                          |
+------+---+----------------------------------------------------------+
| sche | d | dict including execute flag, scheduler fkt type and      |
| dule | i | possible parameters, scheduler has to be included in     |
| r:   | c | torch.optim.lr_scheduler (default: {‘execute’: False})   |
|      | t |                                                          |
+------+---+----------------------------------------------------------+
| num_ | i | number of workers in DataLoaders (default: 10)           |
| work | n |                                                          |
| ers: | t |                                                          |
+------+---+----------------------------------------------------------+
| batc | i | batch size of DataLoaders (default: 64)                  |
| h:   | n |                                                          |
|      | t |                                                          |
+------+---+----------------------------------------------------------+
| outp | s | torch.nn activation fkt at the end of the last layer     |
| ut_a | t | (default: None)                                          |
| ctiv | r |                                                          |
| atio |   |                                                          |
| n:   |   |                                                          |
+------+---+----------------------------------------------------------+

The required hyperparameters are the input and output dimension, as well
as the number of hidden layers with the corresponding number of hidden
units for each layer. Furthermore, some other hyperparameters are
changed in this example. In particular the loss and optimizer function
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


.. parsed-literal::

    /home/pascal/anaconda3/envs/ML/lib/python3.6/site-packages/pytorch_lightning/utilities/distributed.py:23: UserWarning: Metric `ExplainedVariance` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(*args, **kwargs)


In order to allow us to recreate the DataLoader, the necessary
information have to be included in the Model checkpoint. Since the
DataLoader, in its initialization process, aims to load the used data
file, the randomely created DataFrame is saved here as .csv file and the
location is passed in the dataLoader params Namespace “lparams” as
“data_path”. In case the DataLoader has been generated directly from a
file, the data_path is save automacially. To include the lparams
Namespace object, the hparams_update function is used. However, the
lparams Namespace has to be included in an own dict. This is necessary
since the hparams object has a key controller that does not include each
loader key but instead just the lparams key.

.. code:: python

    data_path = 'example_samples.csv'
    
    example_df.to_csv(data_path)  # save DataFrame
    dataLoader.lparams.data_path = data_path  # include data_path in dataLoader lparams
    
    model.hparams_update({'lparams': dataLoader.lparams})

Training and Testing
--------------------

Training and testing is performed using the pl.Trainer. In case of this
example, we just want to control whether our network is fully functional
and that our network training can be continued by saving the model
checkpoint. Therefore the checkpointing callback is active and
max_epochs equals 1. The callback has an own flag and has been modified
so that the epoch is only added to the path if multiple models should be
saved.

.. code:: python

    import pytorch_lightning as pl
    
    from stfs_pytoolbox.ML_Utils import callbacks
    
    checkpointing = callbacks.Checkpointing(filepath='checkpoints/model_example')
    
    trainer = pl.Trainer(max_epochs=1, logger=False, checkpoint_callback=checkpointing)
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())


.. parsed-literal::

    GPU available: False, used: False
    INFO:lightning:GPU available: False, used: False
    No environment variable for node rank defined. Set as 0.
    WARNING:lightning:No environment variable for node rank defined. Set as 0.
    
      | Name                     | Type              | Params
    -----------------------------------------------------------
    0 | activation_fn            | ReLU              | 0     
    1 | loss_fn                  | RelativeMSELoss   | 0     
    2 | explained_variance_train | ExplainedVariance | 0     
    3 | explained_variance_val   | ExplainedVariance | 0     
    4 | explained_variance_test  | ExplainedVariance | 0     
    5 | layers                   | Sequential        | 2 K   
    6 | layers.0                 | Linear            | 128   
    7 | layers.1                 | ReLU              | 0     
    8 | layers.2                 | Linear            | 2 K   
    9 | layers.4                 | Linear            | 130   
    INFO:lightning:
      | Name                     | Type              | Params
    -----------------------------------------------------------
    0 | activation_fn            | ReLU              | 0     
    1 | loss_fn                  | RelativeMSELoss   | 0     
    2 | explained_variance_train | ExplainedVariance | 0     
    3 | explained_variance_val   | ExplainedVariance | 0     
    4 | explained_variance_test  | ExplainedVariance | 0     
    5 | layers                   | Sequential        | 2 K   
    6 | layers.0                 | Linear            | 128   
    7 | layers.1                 | ReLU              | 0     
    8 | layers.2                 | Linear            | 2 K   
    9 | layers.4                 | Linear            | 130   



.. parsed-literal::

    HBox(children=(HTML(value='Validation sanity check'), FloatProgress(value=1.0, bar_style='info', layout=Layout…



.. parsed-literal::

    HBox(children=(HTML(value='Training'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), max…



.. parsed-literal::

    HBox(children=(HTML(value='Validating'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), m…


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(HTML(value='Testing'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), max=…


.. parsed-literal::

    --------------------------------------------------------------------------------
    TEST RESULTS
    {'avg_test_loss': tensor(0.8244),
     'test_ExpVar': tensor(-0.0038),
     'test_loss': tensor(0.8244)}
    --------------------------------------------------------------------------------
    


Model Loading and Training Continuation
---------------------------------------

Lets say that a two stage training is intended wheras in the first stage
the “RelativeMSELoss” and in the second stage the normal “mse_loss” is
used. In this case the model has to be loaded and the dataLoader has to
be regenerated (if not started in the same script). Thereby, model and
dataLoader types have to be known. To further train the pretrained
weights, it is crucial that also the trainer flag
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


.. parsed-literal::

    /home/pascal/anaconda3/envs/ML/lib/python3.6/site-packages/pytorch_lightning/utilities/distributed.py:23: UserWarning: Metric `ExplainedVariance` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(*args, **kwargs)
    GPU available: False, used: False
    INFO:lightning:GPU available: False, used: False
    No environment variable for node rank defined. Set as 0.
    WARNING:lightning:No environment variable for node rank defined. Set as 0.
    
      | Name                     | Type              | Params
    -----------------------------------------------------------
    0 | activation_fn            | ReLU              | 0     
    1 | loss_fn                  | RelativeMSELoss   | 0     
    2 | explained_variance_train | ExplainedVariance | 0     
    3 | explained_variance_val   | ExplainedVariance | 0     
    4 | explained_variance_test  | ExplainedVariance | 0     
    5 | layers                   | Sequential        | 2 K   
    6 | layers.0                 | Linear            | 128   
    7 | layers.2                 | Linear            | 2 K   
    8 | layers.4                 | Linear            | 130   
    INFO:lightning:
      | Name                     | Type              | Params
    -----------------------------------------------------------
    0 | activation_fn            | ReLU              | 0     
    1 | loss_fn                  | RelativeMSELoss   | 0     
    2 | explained_variance_train | ExplainedVariance | 0     
    3 | explained_variance_val   | ExplainedVariance | 0     
    4 | explained_variance_test  | ExplainedVariance | 0     
    5 | layers                   | Sequential        | 2 K   
    6 | layers.0                 | Linear            | 128   
    7 | layers.2                 | Linear            | 2 K   
    8 | layers.4                 | Linear            | 130   



.. parsed-literal::

    HBox(children=(HTML(value='Validation sanity check'), FloatProgress(value=1.0, bar_style='info', layout=Layout…



.. parsed-literal::

    HBox(children=(HTML(value='Training'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), max…



.. parsed-literal::

    HBox(children=(HTML(value='Validating'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), m…


.. parsed-literal::

    


.. parsed-literal::

    /home/pascal/anaconda3/envs/ML/lib/python3.6/site-packages/pytorch_lightning/utilities/distributed.py:23: UserWarning: You're resuming from a checkpoint that ended mid-epoch. This can cause unreliable results if further training is done, consider using an end of epoch checkpoint. 
      warnings.warn(*args, **kwargs)



.. parsed-literal::

    HBox(children=(HTML(value='Testing'), FloatProgress(value=1.0, bar_style='info', layout=Layout(flex='2'), max=…


.. parsed-literal::

    --------------------------------------------------------------------------------
    TEST RESULTS
    {'avg_test_loss': tensor(0.8039),
     'test_ExpVar': tensor(-0.0028),
     'test_loss': tensor(0.8039)}
    --------------------------------------------------------------------------------
    

