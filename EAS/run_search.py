import argparse
import datetime
import logging
import os
import pickle
import sys
import time
import math
import random

import numpy as np
import torch
import torch.cuda as cutorch

# from source.active_search import run_active_search
# from source.sampling import run_sampling
from source.cvrp.model import CVRPModel as CVRP_model
from source.cvrp.read_data import read_instance_pkl as CVRP_read_instance_pkl
from source.cvrp.read_data import read_instance_vrp
from source.cvrp.utilities import augment_and_repeat_episode_data as CVRP__augment_and_repeat_episode_data
from source.cvrp.utilities import get_episode_data as CVRP_get_episode_data
from source.eas_emb import run_eas_emb
from source.eas_lay import run_eas_lay
from source.eas_tab import run_eas_tab
from source.tsp.model import TSPModel as TSP_model
from source.tsp.read_data import read_instance_pkl as TSP_read_instance_pkl
from source.tsp.read_data import read_instance_tsp
from source.tsp.utilities import augment_and_repeat_episode_data as TSP_augment_and_repeat_episode_data
from source.tsp.utilities import get_episode_data as TSP_get_episode_data


def get_config():
    parser = argparse.ArgumentParser(description='Efficient Active Search')

    parser.add_argument('-problem', default="TSP", type=str, choices=['TSP', 'CVRP'])
    parser.add_argument('-method', default="eas-emb", type=str, choices=['eas-emb', 'eas-lay', 'eas-tab'], help="EAS method")
    parser.add_argument('-model_path', default="../pretrained/POMO-TSP/checkpoint-3000-tsp100-instance-norm.pt", type=str, help="Path of the trained model weights")
    parser.add_argument('-instances_path', default="../data/TSP/Size_Distribution/tsp200_rotation.pkl", type=str, help="Path of the instances")
    parser.add_argument('-sol_path', default="../data/TSP/Size_Distribution/concorde/tsp200_rotationoffset0n1000-concorde.pkl", type=str, help="Path of the optimal sol")
    parser.add_argument('-num_instances', default=1000, type=int, help="Maximum number of instances that should be solved")
    parser.add_argument('-instances_offset', default=0, type=int)
    parser.add_argument('-round_distances', default=False, action='store_true', help="Round distances to the nearest integer. Required to solve .vrp instances")
    parser.add_argument('-loc_scaler', default=1.0, type=float, help="The scaler of coordinates to valid range [0, 1]")
    parser.add_argument('-max_iter', default=200, type=int, help="Maximum number of EAS iterations")
    parser.add_argument('-max_runtime', default=100000, type=int, help="Maximum runtime of EAS per batch in seconds")
    parser.add_argument('-batch_size', default=150, type=int)  # Set to 1 for single instance search
    parser.add_argument('-p_runs', default=1, type=int)  # If batch_size is 1, set this to > 1 to do multiple runs for the instance in parallel
    parser.add_argument('-output_path', default="EAS_results", type=str)
    parser.add_argument('-norm', default="instance", choices=['instance', 'batch', 'batch_no_track', 'none'], type=str)
    parser.add_argument('-gpu_id', default=0, type=int)
    parser.add_argument('-seed', default=2023, type=int, help="random seed")

    # EAS-Emb and EAS-Lay parameters
    parser.add_argument('-param_lambda', default=0.0058, type=float)
    parser.add_argument('-param_lr', default=0.0032, type=float)

    # EAS-Tab parameters
    parser.add_argument('-param_alpha', default=0.505, type=float)
    parser.add_argument('-param_sigma', default=8.57, type=float)

    config = parser.parse_args()

    return config


def read_instance_data(config):
    logging.info(f"Reading in instances from {config.instances_path}")

    if config.instances_path.endswith(".pkl"):
        # Read in an instance file that has been created with
        # https://github.com/wouterkool/attention-learn-to-route/blob/master/generate_data.py

        if config.problem == "TSP":
            instance_data = TSP_read_instance_pkl(config.instances_path)
            instance_data = instance_data[config.instances_offset:config.instances_offset + config.num_instances]
            problem_size = instance_data.shape[1]
            instance_data_scaled = (instance_data, None)

        elif config.problem == "CVRP":
            instance_data = CVRP_read_instance_pkl(config.instances_path)
            instance_data = (instance_data[0][config.instances_offset:config.instances_offset + config.num_instances],
                             instance_data[1][config.instances_offset:config.instances_offset + config.num_instances])
            problem_size = instance_data[0].shape[1] - 1

            # The vehicle capacity (here called demand_scaler) is hardcoded for these instances as follows
            # demand_scaler = math.ceil(30 + problem_size / 5) if problem_size >= 20 else 20
            # instance_data_scaled = instance_data[0], instance_data[1] / demand_scaler  # already done in fun(read_instance_pkl)
            instance_data_scaled = instance_data[0], instance_data[1]

    else:
        # Read in .vrp instance(s) that have the VRPLIB format. In this case the distances between customers should be rounded.
        assert config.round_distances

        if config.instances_path.endswith(".vrp") or config.instances_path.endswith(".tsp"):
            # Read in a single instance
            instance_file_paths = [config.instances_path]
        else:
            print("Not supported for Dir now.")
            raise NotImplementedError

        # elif os.path.isdir(config.instances_path):
        #     # or all instances in the given directory.
        #     instance_file_paths = [os.path.join(config.instances_path, f) for f in sorted(os.listdir(config.instances_path))]
        #     instance_file_paths = instance_file_paths[config.instances_offset:config.instances_offset + config.num_instances]

        # Read in the first instance only to determine the problem_size
        if config.instances_path.endswith(".vrp"):
            config.loc_scaler = 1000
            _, locations, _, _ = read_instance_vrp(instance_file_paths[0])
            problem_size = locations.shape[1] - 1
            # Prepare empty numpy array to store instance data
            instance_data_scaled = (np.zeros((len(instance_file_paths), locations.shape[1], 2)),
                                    np.zeros((len(instance_file_paths), locations.shape[1] - 1)))
            # Read in all instances
            for idx, file in enumerate(instance_file_paths):
                # logging.info(f'Instance: {os.path.split(file)[-1]}')
                original_locations, locations, demand, capacity = read_instance_vrp(file)
                instance_data_scaled[0][idx], instance_data_scaled[1][idx] = locations, demand / capacity
        elif config.instances_path.endswith(".tsp"):
            _, locations, _ = read_instance_tsp(instance_file_paths[0])
            problem_size = locations.shape[1]
            # Prepare empty numpy array to store instance data
            instance_data_scaled = (np.zeros((len(instance_file_paths), locations.shape[1], 2)), None)
            # Read in all instances
            for idx, file in enumerate(instance_file_paths):
                # logging.info(f'Instance: {os.path.split(file)[-1]}')
                original_locations, locations, loc_scaler = read_instance_tsp(file)
                instance_data_scaled[0][idx] = locations
            config.loc_scaler = loc_scaler
            config.original_loc = torch.Tensor(original_locations)

    return instance_data_scaled, problem_size


def search(run_id, config):

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'norm': config.norm,
    }
    config.model_params = model_params

    # Creating output directories
    if config.output_path == "":
        config.output_path = os.getcwd()
    now = datetime.datetime.now()
    config.output_path = os.path.join(config.output_path, f"run_{now.year}.{now.month}.{now.day}_{now.hour}{now.minute}{now.second}")
    os.makedirs(os.path.join(config.output_path))

    # Create logger and log run parameters
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Call: {0}".format(' '.join(sys.argv)))

    # Load models
    if config.problem == "TSP":
        model = TSP_model(**model_params).cuda()
    elif config.problem == "CVRP":
        model = CVRP_model(**model_params).cuda()
    else:
        raise NotImplementedError("Unknown problem")
    checkpoint = torch.load(config.model_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    instance_data_scaled, problem_size = read_instance_data(config)

    if config.problem == "TSP":
        get_episode_data_fn = TSP_get_episode_data
        augment_and_repeat_episode_data_fn = TSP_augment_and_repeat_episode_data
    elif config.problem == "CVRP":
        get_episode_data_fn = CVRP_get_episode_data
        augment_and_repeat_episode_data_fn = CVRP__augment_and_repeat_episode_data

    # if config.method == "sampling":
    #     start_search_fn = run_sampling
    # elif config.method.startswith("as"):
    #     start_search_fn = run_active_search
    if config.method.startswith("eas-emb"):
        start_search_fn = run_eas_emb
    elif config.method.startswith("eas-lay"):
        start_search_fn = run_eas_lay
    elif config.method.startswith("eas-tab"):
        start_search_fn = run_eas_tab
    else:
        raise NotImplementedError("Unknown search method")

    if config.batch_size == 1:
        logging.info("Starting single instance search. 1 instance is solved per episode.")
    else:
        assert config.p_runs == 1
        logging.info(f"Starting batch search. {config.batch_size} instances are solved per episode.")

    # Run the actual search
    start_t = time.time()
    perf, best_solutions = start_search_fn(model, instance_data_scaled, problem_size, config, get_episode_data_fn, augment_and_repeat_episode_data_fn)
    runtime = time.time() - start_t

    if config.problem == "CVRP" and not config.instances_path.endswith(".pkl"):
        # For instances with the CVRPLIB format the costs need to be adjusted to match the original coordinates
        perf = np.round(perf * config.loc_scaler).astype('int')
    elif config.problem == "TSP" and not config.instances_path.endswith(".pkl"):
        # For instances with the TSPLIB format the costs need to be adjusted to match the original coordinates
        perf = np.round(perf * config.loc_scaler).astype('int')
        # [!] double-check, we regard best_obj as our final result
        # since the current implementation is inaccurate (e.g, it may even outperform the obj of optimal sol)
        best_sol, best_obj = best_solutions.tolist()[0], 0
        for i in range(problem_size):
            if i == problem_size - 1:
                best_obj += ((config.original_loc[0, best_sol[i]] - config.original_loc[0, best_sol[0]]) ** 2).sum().sqrt().item()
                break
            best_obj += torch.round(((config.original_loc[0, best_sol[i]] - config.original_loc[0, best_sol[i+1]]) ** 2).sum().sqrt()).item()
        print(">> best_obj {} = [{}]".format(best_obj, np.round(best_obj).astype('int')))

    # compute gaps
    if config.instances_path.endswith(".pkl"):
        with open(config.sol_path, 'rb') as f:
            opt_sol = pickle.load(f)[config.instances_offset: config.instances_offset+config.num_instances]  # [(obj, route), ...]
            print(">> Load {} optimal solutions ({}) from {}".format(len(opt_sol), type(opt_sol), config.sol_path))
        assert len(opt_sol) == len(perf)
        gap_list = [(perf[i] - opt_sol[i][0]) / opt_sol[i][0] * 100 for i in range(len(perf))]

        logging.info(f"EAS Method: {config.method}, Seed: {config.seed}")
        logging.info(f"Mean costs: {np.mean(perf)}")
        logging.info(f"Mean gaps: {sum(gap_list)/len(gap_list)}%")
        logging.info(f"Runtime: {runtime}s")
        logging.info("MEM: " + str(cutorch.max_memory_reserved(config.gpu_id) / 1024 / 1024) + "MB")
        logging.info(f"Num. instances: {len(perf)}")

        res = {"EAS_score_list": perf.tolist(), "EAS_gap_list": gap_list}

        pickle.dump(res, open(os.path.join(config.output_path, "./Results_{}.pkl".format(config.method)), 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        logging.info(f"EAS Method: {config.method}, Seed: {config.seed}")
        logging.info(f"Mean costs: {np.mean(perf)}")
        logging.info(f"Runtime: {runtime}s")
        logging.info("MEM: " + str(cutorch.max_memory_reserved(config.gpu_id) / 1024 / 1024) + "MB")
        logging.info(f"Num. instances: {len(perf)}")
        print(">> Solved {}, with sol {}".format(config.instances_path, perf))

    return np.mean(perf)


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    run_id = np.random.randint(10000, 99999)
    config = get_config()
    seed_everything(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_id)

    sol_list, path_list = [], []
    if os.path.isdir(config.instances_path):
        path_list = [os.path.join(config.instances_path, f) for f in sorted(os.listdir(config.instances_path))]
    else:
        path_list = config.instances_path

    for path in path_list:
        config.instances_path = path
        sol = search(run_id, config)
        sol_list.append(sol)
    print(len(sol_list), sol_list)
