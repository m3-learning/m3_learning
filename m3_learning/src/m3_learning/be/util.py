import pyUSID as usid
import h5py
from ..util.h5_util import print_tree
from BGlib import be as belib
import sidpy
import numpy as np


def print_be_tree(path):
    """Utility file to print the Tree of a BE Dataset

    Args:
        path (str): path to the h5 file
    """
    # Opens the translated file
    h5_f = h5py.File(path, "r+")

    # Inspects the h5 file
    usid.hdf_utils.print_tree(h5_f)

    # prints the structure and content of the file
    print("Datasets and datagroups within the file:\n------------------------------------")
    print_tree(h5_f.file)

    print("\nThe main dataset:\n------------------------------------")
    print(h5_f)
    print("\nThe ancillary datasets:\n------------------------------------")
    print(h5_f.file["/Measurement_000/Channel_000/Position_Indices"])
    print(h5_f.file["/Measurement_000/Channel_000/Position_Values"])
    print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Indices"])
    print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Values"])

    print("\nMetadata or attributes in a datagroup\n------------------------------------")
    for key in h5_f.file["/Measurement_000"].attrs:
        print("{} : {}".format(key, h5_f.file["/Measurement_000"].attrs[key]))
        
