Toolbox Concept
===============

The given Toolbox is build on top of
`PyTorch <https://pytorch.org/docs/1.2.0/>`__ (Version: 1.2) and
`PyTorch
Lightning <https://pytorch-lightning.readthedocs.io/en/0.7.6/>`__
(Version: 0.7.6). It should enable you to use Machine Learning in an
easy and understandable way, as well as remove the need to implement
everything on your own. In particular, the Toolbox is constructed with
the aim that for most tasks only a few parameters have to be defined and
you don’t have to worry about how to build a model, how to use GPUs for
training, and so on. However, certain knowledge about Machine Learning
is required in order to properly adjust the parameters.

A typical ML task in the toolbox requires three different parts:
`DataLoader <./getting_started/DataLoader.html>`__,
`Model <./getting_started/Models.html>`__ and
`Trainer <./getting_started/Trainer.html>`__. In order to understand
their usage, we want to consider a tabulated chemistry approach where
the table is replaced by a NN.

1. Firstly, data will be split into look-up variables (=inputs) and the
   thermochemical properties that should be predicted (=outputs).
   Possible pre-processing options such as normalization are performed
   simultaneously. Both tasks are included in the
   `DataLoader <./getting_started/DataLoader.html>`__
2. Secondly, the `ML Model <./getting_started/Models.html>`__ which
   learns the underlying function in the data is constructed by the
   definition of the hyperparameter.
3. Lastly, the PyTorch Lightining
   `Trainer <./getting_started/Trainer.html>`__ is employed to execute
   the “learning” process of the thermochemical properties. The
   `Model <./getting_started/Models.html>`__ and
   `DataLoader <./getting_started/DataLoader.html>`__ are given as
   inputs, together with other flags that can impact training as well as
   testing.

Additionally, components can be defined in order to influence and
individualize each of the previous steps: -
`Callbacks <./getting_started/Callbacks.html>`__ -
`Metrics <./getting_started/Metrics.html>`__ - `Individual
Modules <./getting_started/Individualized_modules.html>`__

The toolbox provides a YAML interface (recommended to use!) for single
and multi model training. The interface is automatically adjusted for
the intended DataLoader and Model (if pre-constructed in the toolbox).
Besides, the toolbox files can be included in individual scripts to
execute any desired task. However, the multi model training is optimized
for the yaml template and has to be modified in order to use a complete
code based implementation. Both strategies can be combined. A detailed
explanation for the YAML interface is provided unter the following
locations: - `YAML single
model <./working_examples/working_example_yaml.html>`__ - `YAML multi
model <./getting_started/MultiModelTraining.html>`__

To use the full potential of the different classes, you are encouraged
to use the different instructions as well as the documentation for the
different scripts. In order to better understand how to work with the
toolbox, working examples are provided. One for each input strategy and
one to explore the usage of a CNN for classification (using the popular
Mnist dataset).

-  `Working Example:
   Code <./working_examples/working_example_code.html>`__
-  `Working Example:
   Yaml <./working_examples/working_example_yaml.html>`__
-  `Working Example:
   Mnist <./working_examples/working_example_mnist.html>`__

--------------

Author: Pascal Roth

E-Mail: roth.pascal@outlook.de
