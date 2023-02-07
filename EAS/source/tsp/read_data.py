import pickle
import numpy as np


def read_instance_pkl(instances_path):
    with open(instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    return np.array(instances_data)


def read_instance_tsp(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    original_locations = np.expand_dims(original_locations, axis=0)
    loc_scaler = original_locations.max()
    # if original_locations.max() <= 10:
    #     loc_scaler = 10
    # elif original_locations.max() <= 100:
    #     loc_scaler = 100
    # elif original_locations.max() <= 1000:
    #     loc_scaler = 100
    # elif original_locations.max() <= 10000:
    #     loc_scaler = 10000
    # elif original_locations.max() <= 100000:
    #     loc_scaler = 100000
    # else:
    #     raise NotImplementedError
    print(">> loc_sclaer: {}".format(loc_scaler))
    locations = original_locations / loc_scaler  # Scale location coordinates to [0, 1]

    # original_locations: unnormalized with shape of (1, n, 2)
    # locations: normalized to [0, 1] with shape of (1, n, 2)
    # original_locations.max(): normalization scale with shape of (1)
    return original_locations, locations, loc_scaler


if __name__ == "__main__":
    read_instance_tsp("../d493.tsp")
