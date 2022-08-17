#!/usr/bin/env python

import os
import json
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


def run(opts):
    # hard-coded
    opts.graph_size = -1  # for variation_type == size
    opts.variation_type = "mix_dist_size"
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
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        opts.epoch_start = epoch_resume + 1

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

    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model_meta = torch.nn.DataParallel(model_meta)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model_meta)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # generate tasks based on task distribution.
    tasks_list = generate_train_task(opts)

    baseline_dict, val_dict = {}, {}
    print("{} tasks in task list: {}".format(len(tasks_list), tasks_list))

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
            break
        print(">> Epoch {}, alpha: {}".format(epoch, alpha))
        for index_task, task in enumerate(tasks_list):
            baseline = baseline_dict[str(task)]
            val_dataset = val_dict[str(task)]
            meta_train_epoch(model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, alpha, task)

        alpha = alpha * opts.alpha_decay

        if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            print('Saving model and state...')
            save_checkpoint(model_meta, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

        # add validation here.
        pass


if __name__ == "__main__":
    run(get_options())
