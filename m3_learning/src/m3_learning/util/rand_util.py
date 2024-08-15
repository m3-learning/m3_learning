import torch
import os
import numpy as np
import random
import re
import fnmatch

def in_list(list_, pattern):
    """
    Checks if any item in the list matches the given pattern.

    Args:
        list_ (list): The list to search in.
        pattern (str): The pattern to match.

    Returns:
        bool: True if there is a match, False otherwise.
    """
    return len(fnmatch.filter(list_, pattern)) != 0

def rand_tensor(min=0, max=1, size=(1)):
    """
    Generates a random tensor within a specified range.

    Args:
        min (float): The minimum value of the tensor.
        max (float): The maximum value of the tensor.
        size (tuple): The size of the random tensor to generate.

    Returns:
        torch.Tensor: The random tensor.
    """
    out = (max - min) * torch.rand(size) + min
    return out

def set_seeds(seed=42):
    """
    Sets the random seeds for reproducibility.

    Args:
        seed (int): The random seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_tuple_names(data):    
    """
    Takes a tuple of variables and returns a list of their names.

    Args:
        data (tuple): The tuple to extract the variable names from.

    Returns:
        list: A list of strings representing the variable names.
    """    
    inner_variable_names = []

    for element in data:
        for inner_name, inner_value in globals().items():
            if inner_value is element and inner_name != "element":
                inner_variable_names.append(inner_name)
                            
    return inner_variable_names

def extract_number(s):
    """
    Extracts a number from a string.

    Args:
        s (str): The string to extract the number from.

    Returns:
        int or float: The extracted number, or None if no number is found.
    """
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
    """
    Saves a list of items to a text file.

    Args:
        lst (list): The list of items to save.
        filename (str): The name of the file to save to.
    """
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')