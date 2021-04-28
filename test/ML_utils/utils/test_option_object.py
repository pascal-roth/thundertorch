import pytest

from thunder_torch.utils.option_class import OptionClass


@pytest.fixture(scope='module')
def create_OptionsObject() -> OptionClass:
    options = OptionClass()
    options.add_key('n_inp', dtype=int, required=True)
    options.add_key('n_out', dtype=int, required=True)
    options.add_key('hidden_layer', dtype=list, required=True)
    options.add_key('output_relu', dtype=bool, mutually_exclusive=['activation'])
    options.add_key('activation', dtype=str, mutually_exclusive=['output_relu'])
    options.add_key('add_key', dtype=str, attr_of='thunder_torch.utils')
    return options


def test_add_key() -> None:
    options = OptionClass()
    options.add_key('some_key', dtype=str, required=True, mutually_exclusive=['other key'],
                    attr_of='thunder_torch.utils', param_dict=True)

    assert options.keylist['some_key']['dtype'] == str
    assert 'some_key' in options.required_keys
    assert options.keylist['some_key']['mutually_exclusive'] == ['other key']
    assert 'some_key' in options.param_dicts
    assert options.keylist['some_key']['attr_of'] == 'thunder_torch.utils'


def test_check_required_keys(create_OptionsObject: OptionClass) -> None:
    input_dict = {'n_inp': 1, 'n_out': 1, 'hidden_layer': [16, 16], 'output_relu': True}
    create_OptionsObject.check_required_keys(input_dict)

    with pytest.raises(AssertionError):
        input_dict = {'n_inp': 1, 'n_out': 1, 'output_relu': True}
        create_OptionsObject.check_required_keys(input_dict)


def test_check_dtype(create_OptionsObject: OptionClass) -> None:
    input_dict = {'n_inp': 1, 'n_out': 1, 'hidden_layer': [16, 16], 'output_relu': True}
    for key, item in input_dict.items():
        create_OptionsObject.check_dtype(key, item)

    with pytest.raises(AssertionError):
        input_dict = {'n_inp': 1, 'n_out': 'str', 'hidden_layer': [16, 16], 'output_relu': True}
        for key, item in input_dict.items():
            create_OptionsObject.check_dtype(key, item)


def test_mutually_exclusive(create_OptionsObject: OptionClass) -> None:
    input_dict_v0 = {'output_relu': True}
    for key in input_dict_v0.keys():
        create_OptionsObject.check_mutually_exclusive(key, input_dict_v0.keys())

    with pytest.raises(AssertionError):
        input_dict_v1 = {'output_relu': True, 'activation': 'relu'}
        for key in input_dict_v1.keys():
            create_OptionsObject.check_mutually_exclusive(key, input_dict_v1.keys())


def test_check_attr_of(create_OptionsObject: OptionClass) -> None:
    input_dict = {'add_key': 'OptionClass'}
    for key, item in input_dict.items():
        create_OptionsObject.check_attr_of(key, item)

    with pytest.raises(AttributeError):
        input_dict = {'some_key': 'some_attr'}
        create_OptionsObject.add_key('some_key', dtype=str, attr_of='thunder_torch.utils')
        for key, item in input_dict.items():
            create_OptionsObject.check_attr_of(key, item)
