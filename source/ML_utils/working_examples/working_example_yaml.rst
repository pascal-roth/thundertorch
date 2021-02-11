Yaml-Single-Model-Interface (Working Example)
=============================================

When using the yaml interface the first step is to create the template
yaml file. This can be done by using the ml_init script. This script
automatically asks which DataLoader and which Model should be used and
copies the yaml template for the single as well as multi model training
in the current working directory.

.. code:: python

    $ ml_init

In this example the LightningFlexMLP and the TabularLoader are chosen so
that the following yaml_file for the single model training is obtained:

config:
  source_files: individual_fct
  deterministic: True

DataLoader:
  type: TabularLoader
  load_DataLoader:
    path: name.pkl or modelXXX.ckpt
  create_DataLoader:
    raw_data_path: samples_name.csv, .txt, .h5, .flut
    features:
    - feature_1
    - feature_2
    - '...'
    labels:
    - label_1
    - label_2
    - '...'
    validation_data:
      load_data:
        path: samples_name.csv, .txt, .h5, .flut
      split_data:
        method: random/ percentage/ explicit
        params: split_params
    test_data:
      load_data:
        path: samples_name.csv, .txt, .h5, .flut
      split_data:
        method: random/ percentage/ explicit
        params: split_params
    save_Loader:
      path: name.pkl

Model:
  type: LightningFlexMLP
  load_model:
    path: name.ckpt
  create_model:
    n_inp: int
    n_out: int
    hidden_layer: '[int, int, int]'
    output_relu: 'bool (default: False)'
    activation: 'str (default: relu)'
  params:
    loss: str (default:mse_loss)
    optimizer:
      type: 'str (default: Adam)'
      params:
        lr: 'float (default: 1.e-3'
    scheduler:
      execute: ' bool (default: False)'
      type: name
      params:
        cooldown: int
        patience: int
        min_lr: float
    num_workers: 'int (default: 10)'
    batch: 'int (default: 64)'

Trainer:
  params:
    gpus: int
    max_epochs: int
    profiler: bool
  callbacks:
  - type: EarlyStopping
    params:
      monitor: val_loss
      patience: int
      mode: min
  - type: ModelCheckpoint
    params:
      filepath: None
      save_top_k: int
  - type: lr_logger
  logger:
  - type: Comet-ml
    params:
      api_key: personal_comet_api
      project_name: str
      workspace: personal_comet_workspace
      experiment_name: name
  - type: Tensorboard
 

It is visible that the yaml is structured in config, DataLoader, Model
and Trainer tree. In the following, the different parts should be
discussed in detail:

-  config tree:

   -  the “source_files” key is used to add individual modules, its
      usage is explained in `Individual Modules
      Tutorial <../getting_started/Individualized_modules.html>`__
   -  the “deterministic” key is used to make training and tesing
      reproducable. If it is selected the random seed of PyTorch and
      Numpy a set to fix value

-  DataLoader

   -  the keys of the DataLoader are unqiue for the DataLoader selected
      which has to be defined in type. Each DataLoader has a
      yml_template function that can be called as static method and that
      prints the basic yml outline
   -  a detailed explanation of the DataLoader properties can be found
      `here <../getting_started/DataLoader.html>`__

-  Model

   -  the keys of the Model are unqiue for the Model selected which has
      to be defined in type. Each Model has a yml_template function that
      can be called as static method and that prints the basic yml
      outline
   -  a detailed explanation of the Model properties can be found
      `here <../getting_started/Models.html>`__

-  Trainer

   -  the PyTorch Lightning Trainer is independent of the used
      DataLoader and Model, its detailed explanation can be found
      `here <../getting_started/Trainer.html>`__

After adjusting it accordingly to the used case, the yaml file is read
and everything is automated by calling the script “trainFlexNN” with the
name of the yaml file:

.. code:: python

    ~$ trainFlexNN TabularLoader_LightningFlexMLP.yaml

Important Properties:
---------------------

-  validation and test data do not need to be defined
-  all keys are transformed to lower case, except after split_data where
   feature inside the data can be defined
