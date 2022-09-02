
import torch
import numpy as np


def get_random_problems(batch_size, problem_size, num_modes=0, distribution='uniform'):
    # uniform distribution problems.shape: (batch, problem, 2)
    if distribution == "uniform":
        problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == "gaussian_mixture":
        generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=num_modes, low=0, high=1)
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


def generate_gaussian_mixture_tsp(dataset_size, problem_size, num_modes=-1, low=0, high=1):
    """
    TSP(N=problem_size, M=num_modes, L=(low, high))
    GMM-9: each mode with N points; overall clipped to the 0-1 square.
    sc: propto stdev of modes arounf the perfect grid; sc1: stdev at each mode.
    Code from "On the Generalization of Neural Combinatorial Optimization Heuristics".
    """
    import scipy
    from scipy import stats
    from numpy.random import default_rng
    from numpy import meshgrid, array
    # print(">> Generating data using Gaussian Mixture.")
    dataset = []
    if num_modes == 0:
        return torch.rand(size=(dataset_size, problem_size, 2))

    for i in range(dataset_size):
        cur_gauss = np.empty([0, 2])
        remaining_elements = problem_size
        modes_done = 0
        sc = 1. / 9.
        sc1 = .045

        rng = default_rng()
        z = array((1., 3., 5.)) / 6
        z = array(meshgrid(z, z))  # perfect grid\n",
        z += rng.uniform(-sc, sc, size=z.shape)  # shake it a bit\n",
        z = z.reshape(2, 9)
        cells_chosen = np.random.choice(9, num_modes, replace=False)

        mu_x_array = []
        mu_y_array = []
        for mode in cells_chosen:
            # grid_x = mode//3
            # grid_y = mode % 3
            mu_x = z[0][mode]
            mu_y = z[1][mode]
            mu_x_array.append(mu_x)
            mu_y_array.append(mu_y)

            elements_in_this_mode = int(remaining_elements / (num_modes - modes_done))
            samples_x = scipy.stats.truncnorm.rvs((low - mu_x) / sc1, (high - mu_x) / sc1, loc=mu_x, scale=sc1,
                                                  size=elements_in_this_mode)
            samples_y = scipy.stats.truncnorm.rvs((low - mu_y) / sc1, (high - mu_y) / sc1, loc=mu_y, scale=sc1,
                                                  size=elements_in_this_mode)
            samples = np.stack((samples_x, samples_y), axis=1)
            cur_gauss = np.concatenate((cur_gauss, samples))
            remaining_elements = remaining_elements - elements_in_this_mode
            modes_done += 1

        data = torch.Tensor(cur_gauss)
        data = data.reshape(problem_size, 2)
        dataset.append(data)

    # print(num_modes, " dataset ", dataset[0])

    return torch.stack(dataset, dim=0)


if __name__ == "__main__":
    data = generate_gaussian_mixture_tsp(dataset_size=10000, problem_size=20, num_modes=1, low=0, high=1)
    print(type(data), data.size())
