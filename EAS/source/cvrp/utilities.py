from source.utilities import augment_xy_data_by_8_fold
from source.TORCH_OBJECTS import Tensor


def get_episode_data(data, episode, batch_size, problem_size):
    node_data = Tensor(data[0][episode:episode + batch_size])
    demand_data = Tensor(data[1][episode:episode + batch_size])

    return node_data, demand_data


def augment_and_repeat_episode_data(episode_data, problem_size, nb_runs, aug_s):
    node_data = episode_data[0]
    demand_data = episode_data[1]

    batch_size = node_data.shape[0]

    depot_xy = node_data[:, [0], :]
    node_xy = node_data[:, 1:, :]
    node_demand = demand_data.reshape(batch_size, problem_size, 1)

    if nb_runs > 1:
        assert batch_size == 1
        depot_xy = depot_xy.repeat(nb_runs, 1, 1)
        node_xy = node_xy.repeat(nb_runs, 1, 1)
        node_demand = node_demand.repeat(nb_runs, 1, 1)

    if aug_s > 1:
        assert aug_s == 8
        # 8 fold Augmented
        depot_xy = augment_xy_data_by_8_fold(depot_xy)
        # aug_depot_xy.shape = (8*batch, 1, 2)
        node_xy = augment_xy_data_by_8_fold(node_xy)
        # aug_node_xy.shape = (8*batch, problem, 2)
        node_demand = node_demand.repeat(aug_s, 1, 1)
        # aug_node_demand.shape = (8*batch, problem, 2)

    return (depot_xy, node_xy, node_demand)
