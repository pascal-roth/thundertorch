Toolbox Concept
===============

The given Toolbox is build on top of
`PyTorch <https://pytorch.org/docs/1.2.0/>`__ and `PyTorch
Lightning <https://pytorch-lightning.readthedocs.io/en/0.7.6/>`__. It
should enable you to use Machine Learning in an easy and understandable
way, as well as remove the need to implement everything on your own. In
particular, the Toolbox is constructed with the aim that for most tasks
only a few parameters have to be defined and you don’t have to worry
about how to build a model, how to use GPUs for training, and so on.
However, certain knowledge about Machine Learning is required in order
to properly adjust the parameters. Thereby, a typical ML task in the
toolbox requires three different parts which have their own detailed
explanation. In order to understand their usage, we want to consider a
tabulated chemistry approach.

1. Firstly, data will be split into look-up variables (=inputs) and
   corresponing thermochemical properties (=outputs). Possible
   pre-processing options such as normalization are performed
   simultaneously. Both methods are included in a
   `DataLoader <./getting_started/DataLoader.html>`__
2. Secondly, the `ML Model <./getting_started/Models.html>`__ itself is
   constructed, hyperparameter have to defined
3. Lastly, the PyTorch Lightining
   `Trainer <./getting_started/Trainer.html>`__ is employed to “learn”
   the thermochemical properties. The
   `Model <./getting_started/Models.html>`__ and
   `DataLoader <./getting_started/DataLoader.html>`__ are given as
   inputs, together other flags that can impact training as well as
   testing.

Construction of the different parts can be done in two different ways
(which also can be combined):

-  by a code implementation

   -  the different models and loaders can be included and used to
      construct any desired script

-  by using the yaml interface

   -  for a single model
   -  for an arbitrary number of models as a queue of parallel processes
      which is explained separately
      `here <./getting_started/MultiModelTraining.html>`__

To use the full potential of the different classes, you are encouraged
to use the different instructions as well as the documentation for the
different scripts. If the provided element does not fit the addressed
task, it is possible to generate your own DataLoaders and Models. For
each, a template is provided and it is recommended to include the
mentioned part since they are addressed when executing the different
task.

In order to better understand how to work the toolbox, working examples
are provided. One for each input strategy and one to explore the usage
of a CNN for classification (using the popular Mnist dataset).

-  `Working Example:
   Code <./working_examples/working_example_code.html>`__
-  `Working Example:
   Yaml <./working_examples/working_example_yaml.html>`__
-  `Working Example:
   Mnist <./working_examples/working_example_mnist.html>`__

--------------

Author: Pascal Roth

E-Mail: roth.pascal@outlook.de
