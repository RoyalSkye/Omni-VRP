#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, seed_everything
from train import clip_grad_norms, tune_and_test
import copy
import pickle
import os


def run(opts):
    # hard-coded
    opts.load_path = "./pretrained/tsp/SIZE/META_10_20_30_50/epoch-99.pt"
    opts.variation_type = "size"

    tune_sequence = []
    epoch = 99999

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    seed_everything(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    print(opts.device)

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is not None, "load path cannot be None!"
    load_path = opts.load_path
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    print("load")
    model_meta = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model_meta = torch.nn.DataParallel(model_meta)

    model_ = get_inner_model(model_meta)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    tasks_list = []
    if opts.variation_type == 'size':
        graph_sizes = [80, 100, 120, 150]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'scale':
        scales = [[0.0, 3.0], [0.0, 5.0], [0.0, 8.0], [0.0, 10.0]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem,task_prop['graph_size'], task_prop['low'], task_prop['high'], "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], task_prop['low'], task_prop['high'], "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'dist':
        for i in [3, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'cap_vrp':
        for i in [20, 50]:
            vrp_capacity = i
            task_prop = {'graph_size': opts.graph_size, 'vrp_capacity': vrp_capacity, 'low': 0, 'high': 1, 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [3, 5, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)
    print("Task list: {}".format(tasks_list))

    baseline_dict, val_dict, fine_tuning_dict = {}, {}, {}
    for task in tasks_list:
        baseline = RolloutBaseline(model_meta, problem, opts, task=task, update_baseline=False)
        baseline_dict[str(task)] = baseline
        print(">> Loading test/fine-tune dataset for task {}".format(task))
        val_dataset = problem.make_dataset(filename='./data/'+opts.variation_type+'/'+opts.problem+'/'+task['test_dataset'], task=task)  # num_samples=opts.test_size,
        # print(len(val_dataset))  # 5000
        val_dict[str(task)] = val_dataset
        fine_tuning_dataset = problem.make_dataset(filename='./data/' + opts.variation_type + '/' + opts.problem + '/' + task['fine_tuning_dataset'], task=task)
        # print(len(fine_tuning_dataset))  # 3000
        fine_tuning_dict[str(task)] = fine_tuning_dataset

    total_reward_tasks = 0
    dict_results_task_sample_iter_wise = {}

    for task in tasks_list:
        task_string = None
        if opts.variation_type == 'dist':
            task_string = task['num_modes']
        if opts.variation_type == 'scale':
            task_string = str(task['low']) + '_' + str(task['high'])
        if opts.variation_type == 'size':
            task_string = task['graph_size']
        if opts.variation_type == 'cap_vrp':
            task_string = task['vrp_capacity']
        if opts.variation_type == 'mix_dist_size':
            task_string = task['num_modes']
        print(">> opts.variation_type:", opts.variation_type)
        print(">> task:", task)
        print(">> task_string:", task_string)

        baseline = baseline_dict[str(task)]
        val_dataset = val_dict[str(task)]
        fine_tuning_dataset = fine_tuning_dict[str(task)]
        dict_results_task_sample_iter_wise[task_string] = {}
        updated_reward = tune_and_test(task, model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, fine_tuning_dataset, dict_results_task_sample_iter_wise[task_string])
        total_reward_tasks += updated_reward

    with open("results_all/test/TEST_" + opts.test_result_pickle_file, 'wb') as handle:
        pickle.dump(dict_results_task_sample_iter_wise, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("EPOCH ID ", opts.load_path)
    avg_rewards_val = total_reward_tasks/len(tasks_list)
    print("Avg reward all tasks after fine tune ", )
    tune_sequence.append(avg_rewards_val)

    for index, x in enumerate(tune_sequence):
        print(index, x.data)


if __name__ == "__main__":
    run(get_options())
