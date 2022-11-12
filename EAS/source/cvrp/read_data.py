import numpy as np
import pickle


def read_instance_pkl(instances_path):
    with open(instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    coord = []
    demands = []
    for instance_data in instances_data:
        coord.append([instance_data[0]])
        coord[-1].extend(instance_data[1])
        coord[-1] = np.array(coord[-1])
        demands.append(np.array(instance_data[2]))

    coord = np.stack(coord)
    demands = np.stack(demands)
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

    return original_locations, locations, demand, capacity
