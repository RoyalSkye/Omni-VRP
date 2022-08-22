#!/usr/bin/env python

import os
import json
import random
import pprint as pp
from datetime import datetime

import torch
import torch.optim as optim

from nets.critic_network import CriticNetwork
from options import get_options
from train import meta_train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, seed_everything, save_checkpoint
from generate_dataset import generate_train_task


def assign_task_group(model_dict):
    """
    The kay idea:
    At the first X epochs, we train one meta-model for all tasks (collect gradient at the same time).
    Then, we cluster tasks into several group (deterministic), and continue training afterwards.
    """
    pass


def run(opts):
    # hard-coded
    opts.shuffle = True
    opts.graph_size = -1
    opts.group = 3
    opts.variation_type = "size"
    opts.baseline_every_Xepochs_for_META = 7
    opts.val_dataset = "../data/size/tsp/tsp100_validation_seed4321.pkl"

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
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        opts.epoch_start = epoch_resume + 1

    # Initialize model
    model_dict, baseline_dict, val_dict = {}, {}, {}
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
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

    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # generate tasks based on task distribution.
    tasks_list = generate_train_task(opts)
    print("{} tasks in task list: {}".format(len(tasks_list), tasks_list))

    for task in tasks_list:
        baseline = RolloutBaseline(model, problem, opts, task=task)
        baseline_dict[str(task)] = baseline
        val_dataset = problem.make_dataset(num_samples=opts.val_size, distribution=opts.data_distribution, task=task)
        val_dict[str(task)] = val_dataset

    alpha = opts.alpha
    start_time = datetime.now()
    grad_dir = {}
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        if (datetime.now() - start_time).total_seconds() >= 24*60*60:
            print(">> Time Out: 24hrs. Training finished {} epochs".format(epoch))
            break
        if opts.shuffle:
            random.shuffle(tasks_list)
        print(">> Epoch {}, alpha: {}".format(epoch, alpha))
        if epoch < 100:
            for index_task, task in enumerate(tasks_list):
                baseline = baseline_dict[str(task)]
                val_dataset = val_dict[str(task)]
                meta_train_epoch(model, baseline, epoch, val_dataset, problem, tb_logger, opts, alpha, task)
        else:
            pass

        alpha = alpha * opts.alpha_decay
        # save checkpoint
        if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            print('Saving model and state...')
            save_checkpoint(model_dict, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))
        # add validation here.
        if opts.val_dataset is not None:
            val_dataset = problem.make_dataset(filename=opts.val_dataset)
            avg_reward = validate(model_meta, val_dataset, opts)
            print(">> Epoch {} avg_cost on TSP100 validation set {}".format(epoch, avg_reward))


if __name__ == "__main__":
    run(get_options())
