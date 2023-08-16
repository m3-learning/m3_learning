import torch
import os
import numpy as np
import random
import re
import fnmatch

def in_list(list_, pattern):
    return len(fnmatch.filter(list_, pattern)) != 0

def rand_tensor(min=0, max=1, size=(1)):
    """ Function that generates random tensor between a range of an arbitrary size
    :param min:  sets the minimum value of the parameter
    :type min: float
    :param max:  sets the maximum value of the parameter
    :type max: float
    :param size: sets the size of the random vector to generate
    :type size: tuple
    :return: random tensor generated
    :rtype: tensor
    """

    out = (max - min) * torch.rand(size) + min
    return out

def set_seeds(seed=42):
    """
    :param seed: random value to set the sequence of the shuffle and random normalization
    :type  seed: int
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_tuple_names(data):    
    """takes a tuple of variables and returns a list of the variable names

    Args:
        data (tuple): tuple to extract the variable names

    Returns:
        list: list of strings of the variable names
    """    
    
    # names for the inner variables
    inner_variable_names = []

    # iterate over the tuple
    for element in data:
        
        # iterate over the global variables
        for inner_name, inner_value in globals().items():
            
            # check if the inner value is the same as the element
            if inner_value is element:
                
                # check if the inner name is not element
                if inner_name != "element":
                    inner_variable_names.append(inner_name)
                            
    return inner_variable_names

def extract_number(s):
    match = re.search(r'\d+\.?\d*', s)
    if match is not None:
        number_str = match.group()
        if '.' in number_str:
            return float(number_str)
        else:
            return int(number_str)
    else:
        return None
    
def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')