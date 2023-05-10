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
        return base.create_group(group)
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
        
def find_groups_with_string(filename, desired_string, start_path="/"):
    """
    Find all groups in an h5 file with a certain string in their name,
    even if they are nested within other groups, and return their full paths.

    Args:
    - filename (str): the name of the h5 file
    - desired_string (str): the string to search for in the group names
    - start_path (str): the path within the h5 file to start the search from (default: "/")

    Returns:
    - A list of full paths to groups that contain the desired string
    """
    group_paths_with_string = []

    # open the h5 file
    with h5py.File(filename, 'r') as file:
        
        # recursively search through the groups in the given path
        def search_groups(path):
            for name, obj in file[path].items():
                # if the object is a group, check if it contains the desired string
                if isinstance(obj, h5py.Group):
                    group_path = f"{path}/{name}"
                    if desired_string in name:
                        group_paths_with_string.append(group_path[1:])
                    # if the group contains subgroups, recursively search them
                    search_groups(group_path)
        
        # start the search from the specified path (or from the root if not specified)
        search_groups(start_path)

    return group_paths_with_string