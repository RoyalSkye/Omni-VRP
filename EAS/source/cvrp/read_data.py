import math
import pickle
import numpy as np


def read_instance_pkl(instances_path):
    with open(instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    coord = []
    demands = []
    for instance_data in instances_data:
        coord.append(instance_data[0])  # depot
        coord[-1].extend(instance_data[1])  # nodes
        coord[-1] = np.array(coord[-1])  # (1 + problem_size, 2)
        demands.append(np.array(instance_data[2]) / instance_data[3])

    coord = np.stack(coord)  # (dataset_size, problem_size+1, 2)
    demands = np.stack(demands)  # (dataset_size, problem_size)

    return coord, demands


def read_instance_vrp(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    original_locations = np.expand_dims(original_locations, axis=0)
    locations = original_locations / 1000  # Scale location coordinates to [0, 1]
    demand = demand[1:, 1:].reshape((1, -1))

    # original_locations: unnormalized with shape of (1, n+1, 2)
    # locations: normalized to [0, 1] with shape of (1, n+1, 2)
    # demand: unnormalized with shape of (1, n)
    # capacity: with shape of (1)
    return original_locations, locations, demand, capacity


if __name__ == "__main__":
    read_instance_vrp("../X-n101-k25.vrp")
