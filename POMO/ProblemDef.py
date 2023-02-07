import os, sys
import math
import glob
import torch
import pickle
import numpy as np
from utils.functions import show, seed_everything, load_dataset, save_dataset


def generate_task_set(meta_params):
    """
    Current setting:
        size: (n,) \in [20, 150]
        distribution: (m, c) \in {(0, 0) + [1-9] * [1, 10, 20, 30, 40, 50]}
        size_distribution: (n, m, c) \in [50, 200, 5] * {(0, 0) + (1, 1) + [3, 5, 7] * [10, 30, 50]}
    """
    if meta_params['data_type'] == "distribution":  # focus on TSP100 with gaussian mixture distributions
        task_set = [(0, 0)] + [(m, c) for m in range(1, 10) for c in [1, 10, 20, 30, 40, 50]]
    elif meta_params['data_type'] == "size":  # focus on uniform distribution with different sizes
        task_set = [(n,) for n in range(20, 151)]
    elif meta_params['data_type'] == "size_distribution":
        dist_set = [(0, 0), (1, 1)] + [(m, c) for m in [3, 5, 7] for c in [10, 30, 50]]
        task_set = [(n, m, c) for n in range(50, 201, 5) for (m, c) in dist_set]
    else:
        raise NotImplementedError
    print(">> Generating training task set: {} tasks with type {}".format(len(task_set), meta_params['data_type']))
    print(">> Training task set: {}".format(task_set))

    return task_set


def get_random_problems(batch_size, problem_size, num_modes=0, cdist=0, distribution='uniform', path=None, problem="tsp"):
    """
    Generate TSP data within range of [0, 1]
    """
    assert problem in ["tsp", "cvrp"], "Problems not support."

    # uniform distribution problems.shape: (batch, problem, 2)
    if distribution == "uniform":
        problems = np.random.uniform(0, 1, [batch_size, problem_size, 2])
        # problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == "gaussian_mixture":
        problems = generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=num_modes, cdist=cdist)
    elif distribution in ["uniform_rectangle", "gaussian", "cluster", "diagonal", "tsplib", "cvrplib"]:
        problems = generate_tsp_dist(batch_size, problem_size, distribution)
    else:
        raise NotImplementedError

    if problem == "cvrp":
        depot_xy = np.random.uniform(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
        node_demand = np.random.randint(1, 10, size=(batch_size, problem_size))  # (unnormalized) shape: (batch, problem)
        demand_scaler = math.ceil(30 + problem_size/5) if problem_size >= 20 else 20
        capacity = np.full(batch_size, demand_scaler)

    # save as List
    if path is not None:
        if problem == "tsp":
            with open(os.path.join(path, "tsp{}_{}.pkl".format(problem_size, distribution)), "wb") as f:
                pickle.dump(problems.tolist(), f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(path, "cvrp{}_{}.pkl".format(problem_size, distribution)), "wb") as f:
                pickle.dump(list(zip(depot_xy.tolist(), problems.tolist(), node_demand.tolist(), capacity.tolist())), f, pickle.HIGHEST_PROTOCOL)  # [(depot_xy, problems, node_demand), ...]

    # return tensor
    if not torch.is_tensor(problems):
        problems = torch.Tensor(problems)
        if problem == "cvrp":
            depot_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_demand), torch.Tensor(capacity)

    if problem == "tsp":
        return problems
    else:
        return depot_xy, problems, node_demand, capacity


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def generate_gaussian_mixture_tsp(dataset_size, graph_size, num_modes=0, cdist=0):
    '''
    Adaptation from AAAI-2022 "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum".
    '''

    def gaussian_mixture(graph_size=100, num_modes=0, cdist=1):
        '''
        GMM create one instance of TSP-100, using cdist
        '''
        from sklearn.preprocessing import MinMaxScaler
        nums = np.random.multinomial(graph_size, np.ones(num_modes) / num_modes)
        xy = []
        for num in nums:
            center = np.random.uniform(0, cdist, size=(1, 2))
            nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
            xy.extend(nxy)
        xy = np.array(xy)
        xy = MinMaxScaler().fit_transform(xy)
        return xy

    if num_modes == 0:  # (0, 0) - uniform
        return np.random.uniform(0, 1, [dataset_size, graph_size, 2])
    elif num_modes == 1 and cdist == 1:  # (1, 1) - gaussian
        return generate_tsp_dist(dataset_size, graph_size, "gaussian")
    else:
        res = []
        for i in range(dataset_size):
            res.append(gaussian_mixture(graph_size=graph_size, num_modes=num_modes, cdist=cdist))
        return np.array(res)


def generate_tsp_dist(n_samples, n_nodes, distribution):
    """
    Generate tsp instances with different distributions: ["cluster", "uniform_rectangle", "diagonal", "gaussian", "tsplib"]
    from "Generative Adversarial Training for Neural Combinatorial Optimization Models".
    """
    print(">> Generating datasets: {}-{}-{}".format(n_samples, n_nodes, distribution))
    if distribution == "cluster":  # time-consuming
        x = []
        for i in range(n_samples):
            print(n_nodes, i)
            loc = []
            n_cluster = np.random.randint(low=2, high=9)
            loc.append(np.random.randint(1000, size=[1, n_cluster, 2]))
            prob = np.zeros((1000, 1000))
            coord = np.concatenate([np.tile(np.arange(1000).reshape(-1, 1, 1), [1, 1000, 1]),
                                    np.tile(np.arange(1000).reshape(1, -1, 1), [1000, 1, 1])], -1)
            for j in range(n_cluster):
                dist = np.sqrt(np.sum((coord - loc[-1][0, j, :]) ** 2, -1))
                dist = np.exp(-dist / 40)
                prob += dist
            for j in range(n_cluster):
                prob[loc[-1][0, j, 0], loc[-1][0, j, 1]] = 0
            prob = prob / prob.sum()
            index = np.random.choice(1000000, n_nodes - n_cluster, replace=False, p=prob.reshape(-1))
            coord = coord[index // 1000, index % 1000]
            loc.append(coord.reshape(1, -1, 2))
            loc = np.concatenate(loc, 1)
            x.append(loc)
        x = np.concatenate(x, 0) / 1000
    elif distribution == "uniform_rectangle":
        data = []
        for i in range(n_samples):
            width = np.random.uniform(0, 1)
            x1 = np.random.uniform(0, 1, [1, n_nodes, 1])
            x2 = np.random.uniform(0.5 - width / 2, 0.5 + width / 2, [1, n_nodes, 1])
            if np.random.randint(2) == 0:
                data.append(np.concatenate([x1, x2], 2))
            else:
                data.append(np.concatenate([x2, x1], 2))
        x = np.concatenate(data, 0)
    elif distribution == "diagonal":
        data = []
        for i in range(n_samples):
            x = np.random.uniform(low=0, high=1, size=(1, n_nodes, 1))
            r = np.random.uniform(low=0, high=1)
            if np.random.randint(4) == 0:
                x = np.concatenate([x, x * r + (1 - r) / 2], 2)
            elif np.random.randint(4) == 1:
                x = np.concatenate([x, (1 - x) * r + (1 - r) / 2], 2)
            elif np.random.randint(4) == 2:
                x = np.concatenate([x * r + (1 - r) / 2, x], 2)
            else:
                x = np.concatenate([(1 - x) * r + (1 - r) / 2, x], 2)
            width = np.random.uniform(low=0.05, high=0.2)
            x += np.random.uniform(low=-width / 2, high=width / 2, size=(1, n_nodes, 2))
            data.append(x)
        x = np.concatenate(data, 0)
    elif distribution == "gaussian":
        data = []
        for i in range(n_samples):
            mean = [0.5, 0.5]
            cov = np.random.uniform(0, 1)
            cov = [[1.0, cov], [cov, 1.0]]
            x = np.random.multivariate_normal(mean, cov, [1, n_nodes])
            data.append(x)
        x = np.concatenate(data, 0)
    elif distribution in ["tsplib", "cvrplib"]:
        file_names = glob.glob("../data/TSP/tsplib/*.tsp") if distribution == "tsplib" else glob.glob("../data/CVRP/cvrplib/Vrp-Set-X/*.vrp")
        data = []
        for file_name in file_names:
            with open(file_name, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].strip().split(":")[0].split(" ")[0] == "DIMENSION":
                        nodes = int(lines[i].strip().split(" ")[-1])
                x = []
                for i in range(len(lines)):
                    if lines[i].strip() == "NODE_COORD_SECTION":
                        for j in range(i + 1, i + nodes + 1):
                            line = [float(n) for n in lines[j].strip().split()]
                            assert j - i == int(line[0])
                            x.append([line[1], line[2]])
                        break
                if len(x) == 0:
                    continue
                x = np.array(x)
                print(x.shape)

                if x.shape[0] < 500:
                    continue
                for i in range(500):
                    index = np.random.choice(x.shape[0], n_nodes, replace=False)
                    x_new = x[index]
                    data.append(x_new.reshape(1, n_nodes, 2))

        x = np.concatenate(data, 0)
        x = x[np.random.permutation(x.shape[0])]
        print(x.shape)
        assert n_samples <= x.shape[0]
        x = x[:n_samples]
        print(x.shape)

    if distribution != "uniform_rectangle":
        x_min, x_max = x.min(1), x.max(1)
        x = x - x_min.reshape(-1, 1, 2)
        x = x / (x_max - x_min).max(-1).reshape(-1, 1, 1)
        x = x + (1 - x.max(1)).reshape(-1, 1, 2) / 2

    np.random.shuffle(x)

    assert x.shape[0] == n_samples
    assert x.shape[1] == n_nodes
    assert x.shape[2] == 2

    return x


if __name__ == "__main__":
    """
    train seed: 1234 
    val seed: 2022
    test seed: 2023
    """
    path = "../data/TSP/Size_Distribution"
    if not os.path.exists(path):
        os.makedirs(path)
    seed_everything(seed=2023)

    # test data for Table 1
    # for s in [200, 300]:
    #     for dist in ["uniform"]:
    #         print(">> Generating TSP instances following {} distribution!".format(dist))
    #         get_random_problems(2000, s, distribution=dist, path=path, problem="tsp")
    for m, c in [(2, 5)]:
        get_random_problems(2000, 200, num_modes=m, cdist=c, distribution="gaussian_mixture", path=path, problem="cvrp")

    # var-size test data
    # for s in [50, 100, 150, 200, 300, 500, 1000]:
    #     print(">> Generating TSP instances of size {}!".format(s))
    #     get_random_problems(15000, s, distribution="uniform", path=path, problem="tsp")

    # data = generate_gaussian_mixture_tsp(dataset_size=1, graph_size=150, num_modes=3, cdist=10)
    # data = load_dataset("../data/TSP/Size_Distribution/tsp200_gaussian_mixture_2_5.pkl")
    # print(data[0])
    # print(type(data), data.size(), data)
    # x, y = [i[0] for i in data[1]], [i[-1] for i in data[1]]
    # x, y = data[0, :, 0].tolist(), data[0, :, -1].tolist()
    # show([x], [y], label=["Gaussian Mixture"], title="TSP200", xdes="x", ydes="y", path="./tsp.pdf")
