import pickle
import numpy as np


def read_instance_pkl(instances_path):
    with open(instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    return np.array(instances_data)
