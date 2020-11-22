Trainer
=======

After having organized the model as LightningModule, the Trainer
automates everything else. It governs training and testing, can decide
whether and when a model is saved, if the losses are logged, how many
epochs to train and so on … In case only default values are used, the
trainer is reduced to the following code:

.. code:: python

    import pytorch_lightning as pl
    
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader=dataLoader.train_dataloader(), val_dataloaders=dataLoader.val_dataloader())
    trainer.test(model, test_dataloaders=dataLoader.test_dataloader())

Parameters
----------

All parameters of the trainer class are defined in the `PyTorch
Lightning
Docs <https://pytorch-lightning.readthedocs.io/en/0.7.6/trainer.html#trainer-flags>`__.
Here the most commonly used are presented:

-  max_epochs: Stop training once this number of epochs is reached
-  gpus: Number of GPUs to train on or which GPUs to train on
-  profiler: profile individual steps during training and assist in
   identifying bottlenecks
-  resume_from_checkpoint: resume training from a specific checkpoint
   pass in the path here
-  fast_dev_run: Runs 1 batch of train, test, and val to find any bugs

Callbacks
---------

PyTorch Lightning has a callback system to execute arbitrary code.
Callbacks should capture NON-ESSENTIAL logic that is NOT required for
your LightningModule to run. A detailed overview of all Callbacks can be
found in the `PyTorch Lightning
Docs <https://pytorch-lightning.readthedocs.io/en/0.7.6/callbacks.html>`__.
The most important callbacks are:

-  EarlyStopping:

   -  Stop training when a monitored quantity has stopped improving.
   -  has its own keyword in trainer class

-  LearningRateLogger

   -  Log learning rate for lr schedulers during training
   -  logger cannot be false and lr scheduler has to be activated

-  Checkpointing

   -  Automatically save model checkpoints during training
   -  !!! Callback name changed from the Lightning Implementation in
      order to allow a checkpoint label without epoch. As a consequence,
      further training can be initialized in the same script without
      entering the name with the corresponding epoch!!!
   -  has an own keyword in the trainer class

Logger
------

It is recommended to use the `Comet
Logger <https://www.comet.ml/site/>`__. In order to use this logger, a
profile has to be made and the code has to be equipped with:

-  api_key: personal api
-  project_name: str
-  workspace: str
-  experiment_name: str

However, the usage of other logger systems is possible. An overview is
given in the `PyTorch Lightning
Docs <https://pytorch-lightning.readthedocs.io/en/0.7.6/loggers.html>`__
