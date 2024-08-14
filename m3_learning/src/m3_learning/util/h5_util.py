import h5py

def print_tree(parent):
    """
    Utility function to print the tree structure of an h5 file.

    Args:
        parent (h5py.Group): The parent group to start printing the tree from.
    """
    print(parent.name)
    if isinstance(parent, h5py.Group):
        for child in parent:
            print_tree(parent[child])
            
def get_tree(parent):
    """
    Utility function to get the tree structure of an h5 file.

    Args:
        parent (h5py.Group): The parent group to start getting the tree from.

    Returns:
        list: A list containing the names of all groups and datasets in the tree.
    """
    tree = []
    tree.append(parent.name)
    if isinstance(parent, h5py.Group):
        for child in parent:
            tree.extend(get_tree(parent[child]))
            
    return tree


def make_group(base, group):
    """
    Utility function to add a group to an h5 file.

    Args:
        base (h5py.File): The base h5 file to add the new group to.
        group (str): The name of the group to add.
    """
    try: 
        return base.create_group(group)
    except:
        print('Could not add group - it might already exist.')
    
def make_dataset(base, dataset, data):
    """
    Utility function to write or overwrite an h5 dataset.

    Args:
        base (h5py.Group): The base group of the h5 file.
        dataset (str): The name of the dataset to create or overwrite.
        data (np.array): The data to store in the dataset.
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
        filename (str): The name of the h5 file.
        desired_string (str): The string to search for in the group names.
        start_path (str): The path within the h5 file to start the search from (default: "/").

    Returns:
        list: A list of full paths to groups that contain the desired string.
    """
    group_paths_with_string = []

    with h5py.File(filename, 'r') as file:
        
        def search_groups(path):
            """
            Recursively search through the groups in the given path.

            Args:
                path (str): The path to search for groups.

            """
            for name, obj in file[path].items():
                if isinstance(obj, h5py.Group):
                    group_path = f"{path}/{name}"
                    if desired_string in name:
                        group_paths_with_string.append(group_path[1:])
                    search_groups(group_path)
        
        search_groups(start_path)

    return group_paths_with_string

def find_measurement(file, search_string, group, list_all=False):
    """
    Find measurements in an h5 file that match a certain string.

    Args:
        file (str): The name of the h5 file.
        search_string (str): The string to search for in the measurement names.
        group (str): The name of the group to search within.
        list_all (bool): Whether to return all matching measurement names or just the first one (default: False).

    Returns:
        str or list: The matching measurement name(s) if found, or None if not found.
    """
    with h5py.File(file, 'r') as f:
        names = []
        for name in f[group]:
            if search_string in name:
                names.append(name)
                
        if len(names) == 1 and not list_all:
            return names[0]
        return names
    return None
