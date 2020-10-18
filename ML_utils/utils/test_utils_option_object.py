import pytest

from stfs_pytoolbox.ML_Utils.utils.utils_option_class import OptionClass


@pytest.fixture(scope='module')
def create_OptionsObject():
    options = OptionClass()
    options.add_key('n_inp', dtype=int, required=True)
    options.add_key('n_out', dtype=int, required=True)
    options.add_key('hidden_layer', dtype=list, required=True)
    options.add_key('output_relu', dtype=bool, mutually_exclusive=['activation'])
    options.add_key('activation', dtype=str, mutually_exclusive=['output_relu'])
    options.add_key('add_key', dtype=str, attr_of=OptionClass)
    return options


def test_add_key():
    options = OptionClass()
    options.add_key('some_key', dtype=str, required=True, mutually_exclusive=['other key'], attr_of=OptionClass,
                    param_dict=True)

    assert options.keylist['some_key']['dtype'] == str
    assert 'some_key' in options.required_keys
    assert options.keylist['some_key']['mutually_exclusive'] == ['other key']
    assert 'some_key' in options.param_dicts
    assert options.keylist['some_key']['attr_of'] == OptionClass


def test_check_required_keys(create_OptionsObject):
    input_dict = {'n_inp': 1, 'n_out': 1, 'hidden_layer': [16, 16], 'output_relu': True}
    create_OptionsObject.check_required_keys(input_dict)

    with pytest.raises(AssertionError):
        input_dict = {'n_inp': 1, 'n_out': 1, 'output_relu': True}
        create_OptionsObject.check_required_keys(input_dict)


def test_check_dtype(create_OptionsObject):
    input_dict = {'n_inp': 1, 'n_out': 1, 'hidden_layer': [16, 16], 'output_relu': True}
    for key, item in input_dict.items():
        create_OptionsObject.check_dtype(key, item)

    with pytest.raises(AssertionError):
        input_dict = {'n_inp': 1, 'n_out': 'str', 'hidden_layer': [16, 16], 'output_relu': True}
        for key, item in input_dict.items():
            create_OptionsObject.check_dtype(key, item)


def test_mutually_exclusive(create_OptionsObject):
    input_dict = {'output_relu': True}
    for key, item in input_dict.items():
        create_OptionsObject.check_mutually_exclusive(key, input_dict.keys())

    with pytest.raises(AssertionError):
        input_dict = {'output_relu': True, 'activation': 'relu'}
        for key, item in input_dict.items():
            create_OptionsObject.check_mutually_exclusive(key, input_dict.keys())


def test_check_attr_of(create_OptionsObject):
    input_dict = {'add_key': 'add_key'}
    for key, item in input_dict.items():
        create_OptionsObject.check_attr_of(key, item)

    with pytest.raises(AssertionError):
        input_dict = {'some_key': 'some_attr'}
        create_OptionsObject.add_key('some_key', dtype=str, attr_of=OptionClass)
        for key, item in input_dict.items():
            create_OptionsObject.check_attr_of(key, item)