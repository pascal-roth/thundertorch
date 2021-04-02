Callbacks
=========

A callback is a self-contained program that can be reused across
projects. The used `Trainer <./Trainer.html>`__ has a callback system to
execute callbacks at the intended time in the training process when
needed. Callbacks should capture NON-ESSENTIAL logic that is NOT
required for your `model <./Models.html>`__ to run. Consequently,
functionality can be extended without polluting the model. A complete
explanation about how to use the callback methods with PyTorch Lightning
(Version: 0.7.6) is given
`here <https://pytorch-lightning.readthedocs.io/en/latest/callbacks.html>`__.
In this documentation all Model Hooks are introduced and explained. A
Model Hook is the time in the training process such as “on_train_end”
where the corresponding part of the callback is executed. A minimal
example of a Callback is:

.. code:: python

    from pytorch_lightning.callbacks import Callback
    
    class MyPrintingCallback(Callback):
    
        def on_init_start(self, trainer):
            print('Starting to init trainer!')
    
        def on_init_end(self, trainer):
            print('trainer is init now')
    
        def on_train_end(self, trainer, pl_module):
            print('do something when training ends')
    
    trainer = Trainer(callbacks=[MyPrintingCallback()])

**Workaround for Callback Limitations due to the used version**

If we take a look at newer PyTorch Lightning versions, some model hooks
have more input arguments such as “batch”, “batch_idx”, and so on. As a
consequence, callbacks can be used for more complex tasks. With the aim
to allow this extension with the used Lightning Version ( which is
necessaary to due to the limitation of the PyTorch Version), the
LightningModelBase of the toolbox is extended by a “hiddens” key.
LightningModelBase is the base script for all models in the toolbox and
contains repeading functionalities and options to construct most layers.
The hiddens key is added to the training, validation and test step and
contains certain parameters such as

.. code:: python

    hiddens = {'inputs': x.detach(), 'preds': y_hat.detach(), 'targets': y.detach()}

If an own model is constructed, training, validation, and test step can
be overwritten and the hidden key adjusted in order to contain more/
different parameters. In a Callback these parameters can the be used for
the indended tasks:

.. code:: python

    class SomeCallback(Callback):
        
        def on_validation_batch_end(self, trainer, pl_module):
            if hasattr(trainer, 'hiddens'):
                preds = trainer.hiddens["preds"]
                targets = trainer.hiddens["targets"]

Pre-Implemented Callbacks
-------------------------

PyTorch Lightning contains certain pre-implemented Callbacks. In the
following the most important ones should be introduced:

-  `Early
   Stopping <https://pytorch-lightning.readthedocs.io/en/0.7.6/callbacks.html#early-stopping>`__

   -  Stop training when a monitored quantity has stopped improving.

.. code:: python

    class pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='auto', strict=True)

-  `Model
   Checkpointing <https://pytorch-lightning.readthedocs.io/en/0.7.6/callbacks.html#model-checkpointing>`__

   -  Automatically save model checkpoints during training.
   -  **Changed in the Toolbox**: if filepath is given and save_top_k=1
      the model is saved under the defined path without adding the epoch
      as it would be done in the lightning implementation. Hence, a
      continous training with e. g. a different loss function can be
      started in one MultiModel Yaml since the exact filepath is known

.. code:: python

    class stfs_pytoolbox.ML_Utils.callbacks.Checkpointing.Checkpointning(filepath=None, monitor='val_loss', verbose=False, save_top_k=1, save_weights_only=False, mode='auto', period=1, prefix='')

-  `Logging of learning
   rates <https://pytorch-lightning.readthedocs.io/en/0.7.6/callbacks.html#logging-of-learning-rates>`__

   -  Log learning rate for lr schedulers during training

.. code:: python

    class pytorch_lightning.callbacks.lr_logger.LearningRateLogger

Metric Callbacks
----------------

The latest versions of PyTorch Lightning include a metrics class as
documentated in detail
`here <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`__.
Again the limitation in the used version makes a certain workaround
necessary in order to use the metrics introduced in Lightning. In
particular the source code has to be copied in the toolbox (a close
introduction is given `here <./Metrics.html>`__). In order to use the
metrics efficiently it is recommended to implement them in a callback.
Hence, the Models are not polluted and metrics can be easily activated.
An example callback is given for the explained_varience metric:

.. code:: python

    from pytorch_lightning.callbacks import Callback
    from stfs_pytoolbox.ML_Utils import metrics
    
    
    class Explained_Variance(Callback):
    
        def on_init_end(self, trainer):
            self.explained_variance_train = metrics.ExplainedVariance()
            self.explained_variance_val = metrics.ExplainedVariance()
            self.explained_variance_test = metrics.ExplainedVariance()
    
        def on_batch_end(self, trainer, pl_module):
            if hasattr(trainer, 'hiddens'):
                inputs = trainer.hiddens["inputs"]
                preds = trainer.hiddens["preds"]
                targets = trainer.hiddens["targets"]
                self.explained_variance_train(preds, targets)
    
        def on_epoch_end(self, trainer, pl_module):
            train_ExpVar = self.explained_variance_train.compute()
            pbar = {'train_ExpVar': train_ExpVar}
            trainer.add_progress_bar_metrics(pbar)
    
        def on_validation_batch_end(self, trainer, pl_module):
            if hasattr(trainer, 'hiddens'):
                preds = trainer.hiddens["preds"]
                targets = trainer.hiddens["targets"]
                self.explained_variance_val(preds, targets)
    
        def on_validation_end(self, trainer, pl_module):
            pbar = {'val_ExpVar': self.explained_variance_val.compute()}
            trainer.add_progress_bar_metrics(pbar)
    
        def on_test_batch_end(self, trainer, pl_module):
            if hasattr(trainer, 'hiddens'):
                preds = trainer.hiddens["preds"]
                targets = trainer.hiddens["targets"]
                self.explained_variance_test(preds, targets)
    
        def on_test_end(self, trainer, pl_module):
            test_ExpVar = self.explained_variance_test.compute()
            pbar = {'test_ExpVar': test_ExpVar}
            trainer.add_progress_bar_metrics(pbar)
