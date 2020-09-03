

def is_list_of_strings(lst):
    """
    Check if given object is a list of strings
    copied from: https://stackoverflow.com/questions/18495098/python-check-if-an-object-is-a-list-of-strings
    basestring was replaced by str as it is no longer available in python 3
    Parameters
    ----------
    lst -  checked list

    Returns
    -------

    """
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False
