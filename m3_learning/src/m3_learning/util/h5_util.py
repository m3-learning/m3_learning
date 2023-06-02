"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import h5py

# define a small function called 'print_tree' to look at the folder tree structure
def print_tree(parent):
    """Utility function to nicely display the tree of an h5 file

    Args:
        parent (h5py): H5 file to print the tree
    """
    print(parent.name)
    if isinstance(parent, h5py.Group):
        for child in parent:
            print_tree(parent[child])


def make_group(base, group):
    """Utility function to add a group onto a h5_file, adds the dependency to not return and error if it already exists.

    Args:
        base (h5py): base h5 file to add new group
        group (string): name of the 
    """
    try: 
        base.create_group(group)
    except:
        print('could not add group - it might already exist')
    
def make_dataset(base, dataset, data):
    """Utility function to write or overwrite an h5 Dataset

    Args:
        base (h5.DataGroup): Base path of the h5 file
        dataset (str): Dataset name to put in the h5 file
        data (np.array): Data to store in the dataset
    """
    try: 
        base[dataset] = data
    except:
        del base[dataset]
        base[dataset] = data
        
