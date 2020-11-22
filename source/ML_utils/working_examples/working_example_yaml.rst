Working Example: Yaml Interface
===============================

When using the yaml interface the first step is to create the template
yaml file. This can be done by calling the yaml_template function of the
intended model, DataLoader, and the Trainer or by using initializer.py.
This function automatically asks which DataLoader and which model should
be used and copies the yaml template as well as the python script to run
the yaml file in the current working directory. In this example the
LightningFlexMLP and the TabularLoader are chosen so that the following
yaml_file is obtained:

.. code:: python

    python ~/pythonToolBox/stfs_pytoolbox/ML_Utils/initializer.py

.. code:: python

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
     

After adjusting it accordingly to the used case, the yaml file is read
and everything is automated by calling the python script
“flexNN_yaml_single.py” that is automatically copied in the same
directory as the yaml template:

.. code:: python

    python flexNN_yaml_single.py -n path_to_yaml.yaml
