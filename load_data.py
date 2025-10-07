import h5py 
import numpy as np 
import os 


def load_h5_to_dict(filename):
    """
    Loads an HDF5 file and converts it into a dictionary.

    Args:
        filename (str): Name of the HDF5 file to load.

    Returns:
        dict: Dictionary loaded from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in f.keys():
            value = f[key][()]
            if isinstance(value, np.ndarray):
                data[key] = value
            else:
                data[key] = value[0]
        return data