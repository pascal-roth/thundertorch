from typing import Any, Union, KeysView, List

from thunder_torch import _logger
from thunder_torch.utils.general import dynamic_imp


class OptionClass:
    """
    Class to control given parameters regarding their included keys, the datatype of each key, whether there are
    mutually exclusive relations and whether the given key should be an attr of another function. Therefore keys are
    added using the add_key function. In order to control the parameters, they should be given as a dict. Nested dict
    structures are possible, however, for every dict that should be controlled an own OptionClass object is necessary.
    Thus the nested structure inside the dict is rebuild as a nested structure of OptionClass objects.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Create OptionClass Object

        Parameters
        ----------
        kwargs          - template: template of the object that is printed when an exception is raised
        """
        super().__init__()
        self.keylist: dict = {}
        self.required_keys: List[str] = []
        self.param_dicts: List[str] = []

        if kwargs.get('template'):
            self.template = 'Follow the template: \n{}'.format(kwargs.pop('template'))
        else:
            self.template = ''

    def add_key(self, key: str, dtype: Any, **kwargs: Any) -> None:
        """
        Add key to object

        Parameters
        ----------
        key             - str name of the key
        dtype           - datatype of the key
        kwargs
            required: bool                      - if key is required to be included in the parameter dict
            mutually_exclusive : str, list      - key(s) that are mutually exclusive with the given key
            param_dict: bool                    - parameter dicts are not investigated in the checker fkt as another
                                                  OptionClass object
            attr_of: str, list                  - key value has to be an attr of one of the given classes/ module
        """
        self.keylist[key] = {'dtype': dtype}
        if kwargs.pop('required', False):
            self.required_keys.append(key)
        if kwargs.get('mutually_exclusive'):
            self.keylist[key]['mutually_exclusive'] = kwargs.pop('mutually_exclusive')
        if kwargs.pop('param_dict', False):
            self.param_dicts.append(key)
        if kwargs.get('attr_of'):
            self.keylist[key]['attr_of'] = kwargs.pop('attr_of')

        if len(kwargs) != 0:
            _logger.warning('Additional/ unexpected kwargs are given!')

    def check_dict(self, input_dict: dict, **kwargs: Any) -> list:
        """
        Check given parameter dict

        Parameters
        ----------
        input_dict      - parameter dict
        kwargs
            input_key: str                      - in a nested structure the key corresponding to the parameter dict

        Returns
        -------
        param_dicts     - list of keys that have param dicts as values and therefore are not investigated in a nested
                          structure
        """
        input_key = kwargs.pop('input_key', 'Key not given!')

        assert bool(input_dict), 'Dict "{}" is empty! {}'.format(input_key, self.template)
        if not all(key in self.keylist.keys() for key in input_dict.keys()):
            wrong_keys = [x for x in input_dict.keys() if x not in self.keylist.keys()]
            assert False, \
                'Input dict keys: "{}" are not included in the accepted key list of the Options Object "{}"'.\
                format(wrong_keys, self.keylist.keys())

        self.check_required_keys(input_dict)

        for key, item in input_dict.items():
            self.check_dtype(key, item)
            if 'mutually_exclusive' in self.keylist[key]:
                self.check_mutually_exclusive(key, input_dict.keys())
            if 'attr_of' in self.keylist[key]:
                self.check_attr_of(key, item)

        return self.param_dicts

    def check_required_keys(self, input_dict: dict) -> None:
        """
        Control if the parameter dicts includes all required keys
        """
        included_keys = input_dict.keys()
        assert all(keys in included_keys for keys in self.required_keys), \
            'Not all required keys are included in dict! Required keys are: {}.{}'.format(self.required_keys,
                                                                                          self.template)

    def check_dtype(self, key: str, item: Any) -> None:
        """
        Control the datatype of the given key
        """
        if not isinstance(self.keylist[key]['dtype'], list):
            self.keylist[key]['dtype'] = [self.keylist[key]['dtype']]
        assert type(item) in self.keylist[key]['dtype'], 'Key "{}" is expected to have dtype(s) "{}", but "{}" was ' \
                                                         'given!'.format(key, self.keylist[key]['dtype'], type(item))

    def check_mutually_exclusive(self, key: str, key_list: Union[list, KeysView[str]]) -> None:
        """
        Check for mutually exclusive relations in the list of keys of the parameter dict
        """
        m_exclusive_keys = self.keylist[key]['mutually_exclusive']

        if not isinstance(m_exclusive_keys, list):
            m_exclusive_keys = [m_exclusive_keys]

        for exclusive_key in m_exclusive_keys:
            assert exclusive_key not in key_list, '"{}" and "{}" are mutually exclusive! {}'.\
                format(exclusive_key, key, self.template)

    def check_attr_of(self, key: str, item: Any) -> None:
        """
        Check if the key value is an attr of the defined class/ module
        """
        if not isinstance(self.keylist[key].get('attr_of'), list):
            self.keylist[key]['attr_of'] = [self.keylist[key].get('attr_of')]

        # check if searched attributes is found in given module list
        module_found = False
        for module in self.keylist[key]['attr_of']:
            try:
                _, _ = dynamic_imp(module, item)
                module_found = True
                break
            except (ImportError, AttributeError):
                pass
        if not module_found:
            raise AttributeError(f"'Function or class '{item}' not found in '{self.keylist[key].get('attr_of')}'")

    @staticmethod
    def checker(input_dict: dict, option_classes: dict) -> None:
        """
        Control the parameters in a nested dict structure

        Parameters
        ----------
        input_dict          - nested dict
        option_classes      - dict that includes all OptionClass objects that are called during the control of the
                              nested input_dict
        """

        def recursion(input_dict: dict, param_dicts: dict, level: int) -> None:
            """
            Recursive function to call the corresponding OptionClass object for every dict in nested structure
            """
            for key, value in input_dict.items():
                if isinstance(value, dict) and key not in param_dicts[f'{level}']:
                    level += 1
                    param_dicts[f'{level}'] = option_classes[key].check_dict(value, input_key=key)
                    recursion(input_dict=value, param_dicts=param_dicts, level=level)
                    level -= 1
                elif isinstance(value, list) and all(isinstance(elem, dict) for elem in value):
                    for list_dict in value:
                        level += 1
                        param_dicts[f'{level}'] = option_classes[key].check_dict(list_dict, input_key=key)
                        level -= 1

        recursion(input_dict=input_dict, param_dicts={'0': ''}, level=0)
