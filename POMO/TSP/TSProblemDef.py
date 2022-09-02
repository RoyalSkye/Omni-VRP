
import torch
import numpy as np


def generate_task_set(meta_params):
    if meta_params['data_type'] == "distribution":  # focus on the TSP100 with different distributions
        task_set = [(m, l) for l in [1, 10, 20, 30, 50] for m in range(1, 1 + meta_params['num_task'] // 5)] + [(0, 0)]
    elif meta_params['data_type'] == "size":  # focus on uniform distribution with different sizes
        task_set = [(n,) for n in range(5, 5 + 5 * meta_params['num_task'], 5)]
    elif meta_params['data_type'] == "size_distribution":
        task_set = [(m, l) for l in [1, 10, 20, 30, 50] for m in range(1, 11)] + [(0, 0)]
        task_set = [(n, m, l) for n in [25, 50, 75, 100, 125, 150] for (m, l) in task_set]
    else:
        raise NotImplementedError
    print(">> Generating training task set: {} tasks with type {}".format(len(task_set), meta_params['data_type']))
    print(">> Training task set: {}".format(task_set))

    return task_set


def get_random_problems(batch_size, problem_size, num_modes=0, cdist=0, distribution='uniform'):
    """
    Generate TSP data within range of [0, 1]
    """
    # uniform distribution problems.shape: (batch, problem, 2)
    if distribution == "uniform":
        problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == "gaussian_mixture":
        problems = generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=num_modes, cdist=cdist)
    else:
        raise NotImplementedError
    return problems


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
        GMM create one instance of TSP-50, using cdist
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

    if num_modes == 0 and cdist == 0:
        return torch.rand(size=(dataset_size, graph_size, 2))
    else:
        res = []
        for i in range(dataset_size):
            res.append(gaussian_mixture(graph_size=graph_size, num_modes=num_modes, cdist=cdist))
        return torch.Tensor(np.array(res))


if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "..")  # for utils
    from utils.functions import show, seed_everything
    seed_everything(seed=1234)

    data = generate_gaussian_mixture_tsp(dataset_size=64, graph_size=100, num_modes=1, cdist=1)
    print(type(data), data.size(), data)
    x, y = data[0, :, 0].tolist(), data[0, :, -1].tolist()
    show([x], [y], label=["Gaussian Mixture"], title="TSP100", xdes="x", ydes="y", path="./tsp.pdf")
