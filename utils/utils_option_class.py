import logging


class OptionClass:
    def __init__(self, **kwargs):
        super().__init__()
        self.keylist = {}
        self.required_keys = []
        self.param_dicts = []

        if kwargs.get('template'):
            self.template = 'Follow the template: \n{}'.format(kwargs.pop('template'))
        else:
            self.template = ''

    def add_key(self, key, dtype, **kwargs):
        self.keylist[key] = {'dtype': dtype}
        if kwargs.pop('required', False): self.required_keys.append(key)
        if kwargs.get('mutually_exclusive'): self.keylist[key]['mutually_exclusive'] = kwargs.pop('mutually_exclusive')
        if kwargs.pop('param_dict', False): self.param_dicts.append(key)
        if kwargs.get('attr_of'): self.keylist[key]['attr_of'] = kwargs.pop('attr_of')

        if len(kwargs) != 0:
            logging.warning('Additional/ unexpected kwargs are given!')

    def check_dict(self, input_dict, **kwargs):
        input_key = kwargs.pop('input_key', 'Key not given!')

        assert bool(input_dict), 'Dict "{}" is empty! {}'.format(input_key, self.template)
        assert all(key in self.keylist.keys() for key in input_dict.keys()), \
            'Input dict keys: "{}" are partly not included in the keylist of the Options Object "{}"'.\
            format(input_dict.keys(), self.keylist.keys())

        self.check_required_keys(input_dict)

        for key, item in input_dict.items():
            self.check_dtype(key, item)
            if 'mutually_exclusive' in self.keylist[key]: self.check_mutually_exclusive(key, input_dict.keys())
            if 'attr_of' in self.keylist[key]: self.check_attr_of(key, item)

        return self.param_dicts

    def check_required_keys(self, input_dict):
        included_keys = input_dict.keys()
        assert all(keys in included_keys for keys in self.required_keys), \
            'Not all required keys are included in dict! Required keys are: {}.{}'.format(self.required_keys,
                                                                                          self.template)

    def check_dtype(self, key, item):
        if not isinstance(self.keylist[key]['dtype'], list): self.keylist[key]['dtype'] = [self.keylist[key]['dtype']]
        assert type(item) in self.keylist[key]['dtype'], 'Key "{}" is expected to have dtype(s) "{}", but "{}" was ' \
                                                         'given!'.format(key, self.keylist[key]['dtype'], type(item))

    def check_mutually_exclusive(self, key, key_list):
        m_exclusive_keys = self.keylist[key]['mutually_exclusive']

        if not isinstance(m_exclusive_keys, list):
            m_exclusive_keys = [m_exclusive_keys]

        for exclusive_key in m_exclusive_keys:
            assert exclusive_key not in key_list, '"{}" and "{}" are mutually exclusive! {}'.\
                format(exclusive_key, key, self.template)

    def check_attr_of(self, key, item):
        if not isinstance(self.keylist[key].get('attr_of'), list):
            self.keylist[key]['attr_of'] = [self.keylist[key].get('attr_of')]

        assert any(hasattr(fct, item) for fct in self.keylist[key]['attr_of']), \
            'Function "{}" not implemented in "{}"!'.format(item, self.keylist[key].get('attr_of'))

    @staticmethod
    def checker(input_dict: dict, option_classes):

        def recursion(input_dict, param_dicts):
            for key, value in input_dict.items():
                if isinstance(value, dict) and key not in param_dicts:
                    param_dicts = option_classes[key].check_dict(value, input_key=key)
                    recursion(input_dict=value, param_dicts=param_dicts)
                elif isinstance(value, list) and all(isinstance(elem, dict) for elem in value):
                    for list_dict in value:
                        param_dicts = option_classes[key].check_dict(list_dict, input_key=key)

        recursion(input_dict=input_dict, param_dicts=[])
