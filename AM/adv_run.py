#!/usr/bin/env python

import os
import json
import pprint as pp
from datetime import datetime

import torch
import torch.optim as optim

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, seed_everything, save_checkpoint


def run(opts):
    # hard-coded
    opts.graph_size = 40  # for variation_type == size
    opts.variation_type = "dist"
    opts.baseline_every_Xepochs_for_META = 7

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    seed_everything(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    # if not opts.no_tensorboard:
    #     tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
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

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model_meta)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # generate tasks based on task distribution.
    tasks_list = []
    """
    if opts.variation_type == 'size':
        graph_sizes = [10, 20, 30, 50]
        if opts.problem == "tsp":
            pass
            # if opts.graph_size_continuous:
            #     graph_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == 'scale':
        scales = [[0, 1], [0, 2], [0, 4]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform',  'variation_type':opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/SCALE_{}_{}-{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], int(task_prop['low']), int(task_prop['high']))
            tasks_list.append(task_prop)
    elif opts.variation_type == 'dist':
            for i in [1, 2, 5]:
                num_modes = i
                task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm',  'variation_type': opts.variation_type}
                task_prop['insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], task_prop['num_modes'])
                tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [1, 2, 4]:
            for cur_graph_size in [20, 30, 50]:
                num_modes = i
                task_prop = {'graph_size': cur_graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
                task_prop['insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], task_prop['num_modes'])
                tasks_list.append(task_prop)
    elif opts.variation_type == 'cap_vrp':
        train_tasks = [int(tsk) for tsk in (opts.train_tasks).split('_')]
        print("train_tasks ", train_tasks)
        for i in train_tasks:
            vrp_capacity = i
            task_prop = {'graph_size': opts.graph_size, 'vrp_capacity': vrp_capacity, 'low':0, 'high':1, 'variation_type': opts.variation_type}
            tasks_list.append(task_prop)
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)
    """
    for i in range(3):
        task_prop = {'graph_size': opts.graph_size, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': 'none'}
        tasks_list.append(task_prop)

    baseline_dict, val_dict = {}, {}
    print("Task list: {}".format(tasks_list))

    for task in tasks_list:
        baseline = RolloutBaseline(model_meta, problem, opts, task=task)
        baseline_dict[str(task)] = baseline
        val_dataset = problem.make_dataset(num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution, task=task)
        val_dict[str(task)] = val_dataset

    alpha = opts.alpha
    start_time = datetime.now()
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        if (datetime.now() - start_time).total_seconds() >= 24*60*60:
            print(">> Time Out: 24hrs. Training finished {} epochs".format(epoch))
        print(">> Epoch {}, alpha: {}".format(epoch+1, alpha))
        for index_task, task in enumerate(tasks_list):
            baseline = baseline_dict[str(task)]
            val_dataset = val_dict[str(task)]
            train_epoch(model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, alpha, task, adv=True)

        alpha = alpha * opts.alpha_decay

        if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            print('Saving model and state...')
            save_checkpoint(model_meta, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

        # add validation here.
        pass


if __name__ == "__main__":
    run(get_options())
