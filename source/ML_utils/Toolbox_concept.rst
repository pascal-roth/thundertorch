Toolbox Concept
===============

The given Toolbox is build on top of PyTorch and PyTorch Lightning. It
should enable to use Machine Learning in an easy and understable way, as
well as remove the need to implement everything on your own. In
particular the Toolbox is constructed with the aim that for most tasks
only a few parameters have to be defined and you don’t have to worry
about how to build a model, how to use GPUs for training and so on.
However, a certain knownledge about Machine Learning is required in
order to properly adjust the parameters. Here only a short overview is
given on how to combine the differents parts of the toolbox. Thereby, a
typical ML tasks in the toolbox requires three different parts which
have their own detailed explanation:

1. The data samples to train and test the network included in a
   `DataLoader <./getting_started/DataLoader.ipynb>`__
2. The `Model <./getting_started/Models.ipynb>`__ itself
3. The `Trainer <./getting_started/Trainer.ipynb>`__ that takes the
   `Model <./getting_started/Models.ipynb>`__ and trains, as well as test it with the
   data included in the `DataLoader <./getting_started/DataLoader.ipynb>`__

Construction of the different parts can be done in two different ways
(which also can be combined):

-  by a code implementation (`Working Example:
   Code <.working_examples/working_example_code.ipynb>`__)

   -  the different models and loaders can be included and used to
      construct any desired script

-  by using the yaml interface

   -  for a single model: (`Working Example:
      Yaml <./working_examples/working_example_yaml.ipynb>`__)
   -  for an arbriary number of models as a queue of parallel processes
      which is explained separately
      `here <./getting_started/MultiModelTraining.ipynb>`__

To use the full potential of the different classes, you are encourage to
use the different instructions as well as the documentation for the
different scripts. If the provided element do not fit the addressed
task, it is possible to generate own DataLoaders and Models. For each a
template is provided and it is recommendet to include the mentioned part
since they are addressed when executing the different parts.
