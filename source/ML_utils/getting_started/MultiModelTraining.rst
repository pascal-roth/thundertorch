Multi-Model-Training
====================

The toolkit provides a special interface for training multiple models
(on the same time). Thereby, the number of CPU/ GPU per model can be
defined individually. The interface is based on the single model yaml
template and thus requires a single yaml template as basis. In the
following the general structure of the interface is discussed and
special flags are introduced. To execute the yaml interface, the
flexNN_yaml_multi.py is used as follows:

.. code:: python

    python flexNN_yaml_multi.py -n input_MultiModelTraining.yaml

General structure
-----------------

For each model an own entry has to be defined. This entry usually starts
with the required definiton of a template. The template is a single
model yaml file which is described in the `Working Example:
Yaml <../working_examples/working_example_yaml.html>`__ and which is
adapted by the other entries. To be more precise, the mentioned keys,
here for example “raw_data_path=’example_samples.csv”, are searched in
the template file and either replaced or added. The resulting yaml file,
where certain keys have been adapted and the other keys copied from the
template, is then used to execute the DataLoader, Model and Training
operation. In list of dicts such as the callbacks list, the type is
required in order to add/ change keys in the params dict.

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

Special Flags
-------------

The special capability of the interface is that an arbitrary number of
model can be trained in a queue where n models are trained parallel. In
order to allow easiyl modify the process, the flags can be defined prior
to the first model

1. Nbr_processes:

   -  number of processes that should be executed in parallel. The
      number is limited by the number of GPUs (if available) or the
      number of CPUs (if GPU not available). As a default the number of
      processes is equal to the number of GPUs (if available) otherwise
      the number of CPUs.

2. GPU_per_model:

   -  number of GPUs each model should be trained on. The number is
      limited by the available GPUs / number of processes. As default
      each model is trained on a single GPU.

3. Model_run:

   -  define which models should be executed, as a default all models
      are used

.. code:: python

    Nbr_processes: int
    GPU_per_model: int
    Model_run:
      - Model001
