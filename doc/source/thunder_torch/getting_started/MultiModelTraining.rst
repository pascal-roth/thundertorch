Multi-Model-Training
====================

The toolkit provides a special interface for training multiple models
(at the same time). Thereby, the number of CPU/ GPU per model as well as
the number of processes executed at the same time can be defined
individually. The interface is based on the single model yaml and thus
requires a single yaml template as basis. The interface has a config
tree and a tree for each model. In the following, the structure of the
model and config tree is introduced.

In order to generate a template for the MultiModelTraining.yaml, execute
the ml_init script

.. code:: python

    ~$ ml_init

To execute the yaml interface, the trainFlexNNmulti script is used:

.. code:: python

    ~$ trainFlexNNmulti input_MultiModelTraining.yaml

Configuration
-------------

The special capability of the interface is that an arbitrary number of
models can be trained in a queue where n models are trained parallel. In
order to allow an easy modification of the process, the flags can be
defined prior to the first model

1. Nbr_processes:

   -  number of processes that should be executed in parallel. The
      number is limited by the number of GPUs (if available) or the
      number of CPUs (if GPU not available). As a default and if
      GPU_per_model as well as CPU_per_model are not defined, the number
      of processes is equal to the number of GPUs (if available),
      otherwise only one process is executed.

2. GPU_per_model:

   -  number of GPUs each model should be trained on. The number is
      limited by the available GPUs and limits the number of processes
      which can be executed in parallel. As a default, each model is
      trained on a single GPU. (Mutually exclusive to CPU_per_model)

3. CPU_per_model:

   -  number of CPUs each model should be trained on. The number is
      limited by the available CPUs and limits the number of processes
      which can be executed in parallel. As a default, one model is
      trained on all available CPUs. (Mutually exclusive to
      GPU_per_model)

4. Model_run:

   -  define which models should be executed, as a default all models
      are used

.. code:: python

    config:
      Nbr_processes: int
      GPU_per_model: int
      CPU_per_model: int
      Model_run:
        - Model001

Model structure
---------------

For each model, an own entry has to be defined. This entry usually
starts with the required definition of a template. The template is a
single model yaml file which is described in the `Working Example:
Yaml <../working_examples/working_example_yaml.html>`__ and which is
adapted by the other entries. To be more precise, the mentioned keys,
here for example ???raw_data_path=???example_samples.csv???, are searched in
the template file and either replaced or added. The resulting yaml file,
where certain keys have been adapted and the other keys copied from the
template, is then used to execute the DataLoader, Model, and Training
operation.

**Important Properties**:

-  In a list of dicts such as the callbacks list, the type is required
   in order to add/ change keys in the params dict.
-  Only keys in the last layer can be changed and the path towards those
   keys has to be included in both, the template yaml and the model
   definition in the MultiModel yaml.

.. code:: python

    Model001:
      Template: single_model.yaml
      DataLoader:
        create_DataLoader:
          raw_data_path: example_samples.csv
          features: [feature_1, feature_2]
          labels: [label_1, label_2]
      Model:
        params:
          loss: mse_loss
      Trainer:
        params:
          max_epochs: 3
        callbacks:
          - type: Checkpointing
            params:
              filepath: checkpoints/try

Convenience Features
--------------------

There are two features implemented to support a quick construction of
the MultiModelTraining:

-  **path abbrevation**: in order to load a model checkpoint the
   directory is sufficient if only one checkpoint is in the directory
-  **expression replacement**: if a model should be saved or loaded,
   often the name of the model (e. g. model001) is used either in the
   path or for the checkpoint name itself. In order to allow a quicker
   copy-paste creation of multiple models, the expression ?????? can be
   used. When the yaml file is read, this expression will be replaced by
   the model name

   -  ATTENTION: keep in mind that the model name **is not transformed
      to lower letters** (in opposite of all other keys), this can be
      important if the model name should be used in features or labels
      since there occurance in the raw data will most likely be case
      sensitive so that the right still can be found

In the following example, the only model in the directory
???checkpoints/model001??? will be loaded and saved under the path
???checkpoints/model001/model001_conti???:

.. code:: python

    Model001:
      Template: single_model.yaml
      DataLoader:
        create_DataLoader:
          raw_data_path: example_samples.csv
          features: [feature_1, feature_2]
          labels: [label_1, label_2]
      Model:
        load_model:
          path: checkpoints/<model_name>
        params:
          loss: mse_loss
      Trainer:
        params:
          max_epochs: 3
          resume_from_checkpoint: checkpoints/<model_name>
        callbacks:
          - type: Checkpointing
            params:
              filepath: checkpoints/<model_name>/<model_name>_conti
