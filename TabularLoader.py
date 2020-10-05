#######################################################################################################################
# Class including training, validation and test data
#######################################################################################################################

# import packages
import pandas as pd
import random
import numpy as np
import warnings


class TabularLoader:
    def __init__(self, df_samples):
        self.samples_train = df_samples
        self.samples_val = None
        self.samples_test = None

    # validation_data #################################################################################################
    def add_val_data(self, df_val_samples):
        assert isinstance(df_val_samples, pd.DataFrame), 'Tranform data in pandas dataframe!'  # TODO: so lösen oder auch verschiedene Möglichkeiten die Daten zu laden ?
        self.samples_val = df_val_samples

    def val_split(self, method, **kwargs):
        val_size = kwargs.pop('val_size', 0.2)
        val_params = kwargs.pop('val_params', {'T_0': 0.2})
        split_method = kwargs.pop('split_method', 'percentage')

        if method == 'random':
            assert val_size < 1, 'Percentage exceeds 100%!'
            assert isinstance(val_size, float), 'Val_size must be float in range 0 to 1!'
            self.samples_val = self.samples_train.sample(frac=val_size, random_state=1)
            self.samples_train = self.samples_train.drop(index=self.samples_val.index)

        elif method == 'sample':
            self.samples_val = pd.DataFrame([])

            for key, value in val_params.items():

                key_options = self.samples_train['{}'.format(key)].drop_duplicates()

                if split_method == 'percentage':  # TODO: im Moment noch für alle features einheitlich, falls nötig noch anpassen, dass für jedes feature einzeln auswählbar
                    assert value < 1, 'Percentage exceeds 100%!'
                    key_list = random.choices(key_options.values, k=int(np.round(value * len(key_options))))
                    assert len(key_list) > 0, 'Percentage to low that one value of {} is selected'.format(key)

                elif split_method == 'explicit' and isinstance(value, (float, int)):
                    key_list = [value]
                    assert value in key_options.values, 'Selected value: {} is not included included in feature {}'.\
                        format(value, key)

                elif split_method == 'explicit' and isinstance(value, list):
                    key_list = value
                    for i in range(len(value)):
                        assert value[i] in key_options.values, 'Selected value: {} is not included included in ' \
                                                               'feature {}'.format(value[i], key)

                elif split_method == 'explicit':
                    raise KeyError('{} is not a valid input type for split_method "explicit", valid types are: float '
                                   'and int!'.format(type(value)))

                else:
                    raise NameError('{} is no valid split method, chose between "percentage" and "explicit"!'.
                                    format(split_method))

                self.samples_train = self.samples_train.rename(columns={'{}'.format(key): 'feature'})

                for i, key_value in enumerate(key_list):
                    self.samples_val = pd.concat((self.samples_val, self.samples_train[self.samples_train.feature == key_value]), axis=0)
                    self.samples_train = self.samples_train[self.samples_train.feature != key_value]

                self.samples_train = self.samples_train.rename(columns={'feature': '{}'.format(key)})
                self.samples_val = self.samples_val.rename(columns={'feature': '{}'.format(key)})

        else:
            raise NameError('{} is no valid method, choose between "random" and "sample"!'.format(method))

        if len(kwargs) != 0:
            warnings.warn('Additional, unexpected kwargs are given! Only expected args are: '
                          '"method", "val_size", "split_method", "val_params"')

    # test_data #######################################################################################################
    def add_test_data(self, df_test_samples):
        assert isinstance(df_test_samples, pd.DataFrame), 'Tranform data in pandas dataframe!'  # TODO: so lösen oder auch verschiedene Möglichkeiten die Daten zu laden ?
        self.samples_test = df_test_samples

    def test_split(self, method, **kwargs):
        test_size = kwargs.pop('test_size', 0.2)
        test_params = kwargs.pop('test_params', {'T_0': 0.05})
        split_method = kwargs.pop('split_method', 'percentage')

        if method == 'random':
            assert test_size < 1, 'Percentage exceeds 100%!'
            assert isinstance(test_size, float), 'Val_size must be float in range 0 to 1!'
            self.samples_test = self.samples_train.sample(frac=test_size, random_state=1)
            self.samples_train = self.samples_train.drop(index=self.samples_test.index)

        elif method == 'sample':
            self.samples_test = pd.DataFrame([])

            for key, value in test_params.items():

                key_options = self.samples_train['{}'.format(key)].drop_duplicates()

                if split_method == 'percentage':  # TODO: im Moment noch für alle features einheitlich, falls nötig noch anpassen, dass für jedes feature einzeln auswählbar
                    assert value < 1, 'Percentage exceeds 100%!'
                    key_list = random.choices(key_options, k=int(value * len(key_options)))
                    assert len(key_list) > 0, 'Percentage to low that one value of {} is selected'.format(key)

                elif split_method == 'explicit' and isinstance(value, (float, int)):
                    key_list = [value]
                    assert value in key_options.values, 'Selected value: {} is not included included in feature {}'. \
                        format(value, key)

                elif split_method == 'explicit' and isinstance(value, list):
                    key_list = value
                    for i in range(len(value)):
                        assert value[i] in key_options.values, 'Selected value: {} is not included included in ' \
                                                               'feature {}'.format(value[i], key)

                elif split_method == 'explicit':
                    raise KeyError('{} is not a valid input type for split_method "explicit", valid types are: float '
                                   'and int!'.format(type(value)))

                else:
                    raise NameError('{} is no valid split method, chose between "percentage" and "explicit"!'.
                                    format(split_method))

                self.samples_train = self.samples_train.rename(columns={'{}'.format(key): 'feature'})

                for i, key_value in enumerate(key_list):
                    self.samples_test = pd.concat((self.samples_test, self.samples_train[self.samples_train.feature == key_value]), axis=0)
                    self.samples_train = self.samples_train[self.samples_train.feature != key_value]

                self.samples_train = self.samples_train.rename(columns={'feature': '{}'.format(key)})
                self.samples_test = self.samples_test.rename(columns={'feature': '{}'.format(key)})

        else:
            raise NameError('{} is no valid method, choose between "random" and "sample"!'.format(method))

        if len(kwargs) != 0:
            warnings.warn('Additional, unexpected kwargs are given! Only expected args are: '
                          '"method", "test_size", "split_method", "test_params"')

    # save and load TabluarLoader object ##############################################################################
    def save(self):
        pass  # TODO: add function to save TabluarLoader object if useful (useful when multiple models are trained with same data)

    def load(self):
        pass

    # classmethods ####################################################################################################
    @classmethod
    def read_from_flut(cls, file): # TODO: Einzelheiten über Dataformat heraufinden !
        print('not implemented yet -> flut datatype unknown')

    @classmethod
    def read_from_h5(cls, file):
        store = pd.HDFStore(file)
        keys = store.keys()
        assert len(keys) == 1, "There must be only one key stored in pandas.HDFStore in '{}'!".format(file)
        df_samples = store.get(keys[0])
        store.close()
        return cls(df_samples)

    @classmethod
    def read_from_csv(cls, file):
        df_samples = pd.read_csv(file)
        return cls(df_samples)
