DataLoader
==========

DataLoaders are a central element when working with NN. They provide an
iterable over the given dataset. This toolbox has pre-implemented
DataLoaders for different kinds of datasets. However, own DataLoaders
can be created by following the basic structure of the
DataLoaderTemplate.py. Since the DataLoader is passed directly to the
Lightning Trainer class when the model is trained/ tested, the necessary
preprocessing steps have to be included in the DataLoader. Furthermore,
the DataLoader has to consist of data for training, validation, and
testing. If those three datasets are not defined prior to the training,
a default split is performed.

Initialization Methods
----------------------

A DataLoader can be initialized by a direct code implementation or by
using the yaml structure. Thereby, each DataLoader has a yaml template
saved as a staticmethod. In order to employ the yaml file, the utils
functions can be used. The check_argsLoader(args_yaml) function is not
mandatory, however, the usage is recommended in order to detect possible
errors and secure that the intended output is provided. In the
following, the different approaches are demonstrated using the
TabularLoader

.. code:: python

    # direct code implementation
    
    import numpy as np
    import pandas as pd
    from stfs_pytoolbox.ML_Utils import loader
    
    example_df = pd.DataFrame(np.random.rand(10000, 5))
    example_df.columns = ['feature_1', 'feature_2', 'feature_3', 'label_1', 'label_2']
    
    features = ['feature_1', 'feature_2', 'feature_3']
    labels = ['label_1', 'label_2']
    
    dataLoader = loader.TabularLoader(example_df, features, labels)

.. code:: python

    # using the yaml_template
    
    import yaml
    from stfs_pytoolbox.ML_Utils.utils import *
    
    args_yaml = parse_yaml('path.yaml')
    check_argsLoader(args_yaml['dataloader'])
    dataLoader = get_dataLoader(args_yaml['dataloader'])

DataLoader Classmethods
-----------------------

Next to the direct initialization as seen above, DataLoader can be
created directly from different files or by using the information saved
in a model checkpoint. For both cases certain constraints have to be
known:

-  Read from a file

   -  supported file datatypes: .csv, .txt, .h5, .ulf
   -  there has to be only one key stored in the HDFStore

-  Read from checkpoint

   -  !!! The parameters of the Loader have to be included in the
      hparams of the model. Therefore, the model function
      “update_hparams” has to be executed as follows:
      “model.hparams_update(update_dict={‘lparams’:
      dataLoader.lparams})” !!!
   -  the data path to training, validation and test dataset (in case
      the validation and/ or test set are individual loaded datasets and
      not split from the training set) has to be the same as the one
      when the DataLoader has been created in the first place

In the following the two approaches are shown using the TabularLoader:

.. code:: python

    # read from file
    
    from stfs_pytoolbox.ML_Utils import loader
    
    features = ['feature_1', 'feature_2', '...']
    labels = ['label_1', 'label_2', '...']
    
    dataLoader = loader.TabularLoader.read_from_file('file_path.csv', features, labels)

.. code:: python

    # read from checkpoint
    
    from stfs_pytoolbox.ML_Utils import loader
    
    dataLoader = loader.TabularLoader.read_from_checkpoint('file_path.ckpt')

Save and Load DataLoader
------------------------

Besides, it is possible to save and load DataLoader. Therefore, the
pickle data format is chosen. The corresponding functions are called
“save” and “load” with the file_path.pkl as only input:

.. code:: python

    # save dataLoader
    dataLoader.save('file_path.pkl')
    
    # load dataLoader
    dataLoader = loader.TabularLoader.load('file_path.pkl')

Validation and Test Dataset
---------------------------

Prior to the use of the DataLoader an input of the pl.Trainer class, the
DataLoader has to include training, validation, and test data set. These
datasets can be obtained in two different ways. One way is to load
individual datasets for validation and/ or testing. Thereby, the loading
can be performed when the Loader is initialized or afterward by calling
the functions “add_val_data” / “add_test_data”. The same datatypes as
for the training data are supported (.csv, .txt, .h5). An example using
the TabularLoader is as follows:

.. code:: python

    from stfs_pytoolbox.ML_Utils import Loader
    
    # load validation and test data within initialization
    dataLoader = loader.TabularLoader.read_from_file('file_path.csv', features, labels, val_path='path_to_val_data.csv', 
                                                     test_path='path_to_test_data.csv')
    
    # load data after initialization
    dataLoader = loader.TabularLoader.read_from_file('file_path.csv', features, labels)
    dataLoader.add_val_data('path_to_val_data.csv')
    dataLoader.add_test_data('path_to_test_data.csv')

In case validation and test data are not individual datasets, they have
to be separated from the training set. The toolbox provides three
different approaches to fulfill the separation. These are:

-  a random approach (‘method’: ‘random’, ‘params’: float):

   -  a certain percentage of samples is taken randomly

-  a percentage approach (‘method’: ‘percentage’, ‘params’:
   {‘’feature_1’: float, ‘feature_2’: float, …}):

   -  Split the data by extracting the different values of a feature and
      randomly pick a certain percentage of it. All samples where the
      feature is equal to one of those values are extracted into x_split
      / y_split. However, if the feature has a different value for each
      sample, the method is equal to random. Furthermore, the size of
      x_split / y_split can differ from the percentage of values taken.
      In split_params the percentage can be defined for an arbitrary
      number of features.

-  an explicit appraoch (‘method’: ‘explicit’, ‘params’: {‘’feature_1’:
   [value_1, value_2], ‘feature_2’: [value_1, value_2], …}):

   -  Split data according to explicit values of the different features.
      It is possible to define an arbitrary number of values for the
      different features.

Splitting the training data can be performed either by initializing the
DataLoader object or by calling the functions “val_split”/ “test_split”.
In the following an example including all functions using the
TabularLoader is shown:

.. code:: python

    from stfs_pytoolbox.ML_Utils import Loader
    
    # load validation and test data within initialization
    dataLoader = loader.TabularLoader.read_from_file('file_path.csv', features, labels, val_split={'method': 'random', 'params': 0.2}, 
                                                     test_split={'method': 'percentage', 'params': {'feature_1': 0.2}})
    
    # load data after initialization
    dataLoader = loader.TabularLoader.read_from_file('file_path.csv', features, labels)
    dataLoader.val_split(method='random', params=0.2})
    dataLoader.test_split(method='explicit', params={'feature_1': ['value_1', 'value_2']})
