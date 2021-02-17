Individual Modules
==================

Machine Learning is a very broad field and can be used in multiple
occasions. Consequently, the provided elements of the Toolbox may be
insufficient for the addressed tasks and an indivdual implementation of
a model, a dataLoader, a callback or certain functions might be
necessary. The toolbox allows this extension without the need to change
the source code in any way. In the following, the different modules are
introduced where individual extensions are possible, followed by a
tutorial on how to add individual modules.

Expandable modules
------------------

-  models

   -  pre-implemented source: ‘stfs_pytoolbox.ML_Utils.models’
   -  for new models please keep attention to the model template
      (stfs_pytoolbox.ML_Utils.models._LightningModelTemplate.LightningModelTemplate)
      and include the functions mentioned there since they will be used
      when in training

-  DataLoader

   -  pre-implemented source: ‘stfs_pytoolbox.ML_Utils.loader’
   -  for new models please keep attention to the DataLoader template
      (stfs_pytoolbox.ML_Utils.loader._DataLoaderTemplate.DataLoaderTemplate)
      and include the functions mentioned there

-  callbacks

   -  pre-implemented source: ‘pytorch_lightning.callbacks’,
      ‘stfs_pytoolbox.ML_Utils.callbacks’
   -  tutorial on how to construct a callback can be found
      `here <./Callbacks.html>`__

-  activation function

   -  pre-implemented source:
      `‘torch.nn’ <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`__

-  loss function

   -  pre-implemented source:
      `‘torch.nn’ <https://pytorch.org/docs/stable/nn.html#loss-functions>`__,
      ‘stfs_pytoolbox.ML_Utils.models’

-  optimizer

   -  pre-implemented source:
      `‘torch.optim’ <https://pytorch.org/docs/stable/optim.html>`__

-  learning rate scheduler

   -  pre-implemented source:
      `‘torch.optim.lr_scheduler’ <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`__

How to add an individual module?
--------------------------------

Let’s say we want to add a new loss function which computes the relative
MSELoss (function implemented in “stfs_pytoolbox.ML_Utils.models.\_
losses.RelativeMSELoss”). We have initialized the project using ml_init
in our current working directory. In order to add an individual loss
function, we have to pack it in a module (directory with “\_\_ init \_\_
.py”. Therefore, the following example structure will be implemented:

.. code:: python

    .
    +-- TabularLoader_LightningFlexMLP.yaml
    +-- MultiModelTraining.yaml
    +-- individual_module
    |   +-- __init__.py
    |   +-- myloss.py

With the \_\_ init \_\_ .py as follows:

.. code:: python

    from .myloss import RelativeMSE
    
    __all__ == ['RelativeMSE']

And myloss.py with the defintion of the class:

.. code:: python

    class RelativeMSE(nn.Module):
        def __init__(self, ...):
            ...
        
        def forward(self, y, t):
            ...
        

If now the path to our indvidual module should be added to the list of
sources for the loss function,
stfs_pytoolbox.ML_Utils.utils.training.train_config is used. This
function automatically adds the given module(s) to all expandable
modules. If the YAML template is used, the module path has to be added
to the key “source_files” in the config tree. An example for both input
strategies is given here:

.. code:: python

    from stfs_pytoolbox.ML_Utils.utils import train_config
    
    argsConfig = {'source_files': 'individual_module'}
    train_config(argsConfig)
    
    # then initialize model with the loss_fn='RelativeMSE' and start training

.. code:: python

    config:
      source_files: individual_module
      
    ...
    
    Model:
      ...
      params:
        loss: RelativeMSE
        ...
