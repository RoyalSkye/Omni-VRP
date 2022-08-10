from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


def generate_GM_tsp_data_grid(dataset_size, graph_size, num_modes=-1, low=0, high=1):
    """
    GMM-9: each mode with N points; overall clipped to the 0-1 square.
    sc: propto stdev of modes arounf the perfect grid; sc1: stdev at each mode.
    Code from "On the Generalization of Neural Combinatorial Optimization Heuristics".
    """
    from numpy.random import default_rng
    from numpy import meshgrid, array
    print("num modes ", num_modes)
    dataset = []

    for i in range(dataset_size):
        cur_gauss = np.empty([0, 2])
        remaining_elements = graph_size
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
        data = data.reshape(graph_size, 2)
        dataset.append(data)

    print(num_modes, " dataset ", dataset[0])

    return dataset


def generate_tsp_data_mg(dataset_size, graph_size):
    '''
    formal test setting, generate GMM TSP-50 data (number dataset_size). every part dataset_size//12
    Code from AAAI-2022 "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum".
    '''

    def mg(cdist=100, graph_size=50):
        '''
        GMM create one instance of TSP-50, using cdist
        '''
        from sklearn.preprocessing import MinMaxScaler
        nc = np.random.randint(3, 7)
        nums = np.random.multinomial(graph_size, np.ones(nc) / nc)
        xy = []
        for num in nums:
            center = np.random.uniform(0, cdist, size=(1, 2))
            nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
            xy.extend(nxy)

        xy = np.array(xy)
        xy = MinMaxScaler().fit_transform(xy)
        return xy

    pern = [dataset_size // 11] * 10 + [dataset_size-dataset_size//11*10]
    res = []
    # uni = np.random.uniform(size=(dataset_size - pern * 11, graph_size, 2))
    # res.append(uni)
    for i, cdist in enumerate([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        # GMM create a batch size instance of TSP-50, using cdist
        xy_ = []
        for j in range(pern[i]):
            xy_.append(mg(cdist, graph_size))
        res.append(np.array(xy_))
    res = np.concatenate(res, axis=0)

    return res


def generate_uniform_tsp_data(dataset_size, graph_size, low=0, high=1):
    return [torch.FloatTensor(graph_size, 2).uniform_(low, high) for i in range(dataset_size)]


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None, task=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]  # Sample points randomly in [0, 1] square
            if task['variation_type'] == 'size':
                self.data = generate_uniform_tsp_data(num_samples, task['graph_size'], task['low'], task['high'])
            if task['variation_type'] == 'scale':
                self.data = generate_uniform_tsp_data(num_samples, task['graph_size'], task['low'], task['high'])
            if task['variation_type'] == 'dist':
                self.data = generate_GM_tsp_data_grid(num_samples, task['graph_size'], task['num_modes'])
            if task['variation_type'] == 'mix_dist_size':
                self.data = generate_GM_tsp_data_grid(num_samples, task['graph_size'], task['num_modes'])

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
