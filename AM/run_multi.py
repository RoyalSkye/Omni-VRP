#!/usr/bin/env python

import json
import tqdm
import pprint as pp

import torch
import torch.optim as optim
import os
from options import get_options
from torch.utils.data import DataLoader
from train import train_epoch, get_inner_model, clip_grad_norms, get_hard_samples
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel, set_decode_type
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, seed_everything, save_checkpoint, move_to
from generate_dataset import generate_train_task
import datetime


def run(opts):
    # hard-coded
    opts.graph_size = -1  # for variation_type == size
    opts.variation_type = "mix_dist_size"
    update_task = False  # update AM by batch (default) or task (implementation of "On the Generalization of Neural Combinatorial Optimization Heuristics")
    eps = 0
    # opts.baseline_every_Xepochs_for_META = 40  # set to default value for multi-AM / oracle-AM

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
    model_common = model_class(
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
    model_ = get_inner_model(model_common)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # generate tasks based on task distribution.
    tasks_list = generate_train_task(opts)

    baseline_dict, val_dict = {}, {}
    print("{} tasks in task list: {}".format(len(tasks_list), tasks_list))

    for task in tasks_list:
        baseline = RolloutBaseline(model_common, problem, opts, task=task)
        baseline_dict[str(task)] = baseline
        val_dataset = problem.make_dataset(num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution, task=task)
        val_dict[str(task)] = val_dataset

    optimizer_common = optim.Adam(model_common.parameters(), opts.lr_model)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_common, lambda epoch1: opts.lr_decay ** epoch1)
    start_time = datetime.now()

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        if (datetime.now() - start_time).total_seconds() >= 24 * 60 * 60:
            print(">> Time Out: 24hrs. Training finished {} epochs".format(epoch))
            break
        if not update_task:
            for task in tasks_list:
                baseline = baseline_dict[str(task)]
                val_dataset = val_dict[str(task)]
                train_epoch(model_common, optimizer_common, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, task, eps=eps)
        else:
            for task in tasks_list:
                baseline = baseline_dict[str(task)]
                val_dataset = val_dict[str(task)]

                # update by task
                epoch_size = opts.batch_size * opts.k_tune_steps
                training_dataset = baseline.wrap_dataset(problem.make_dataset(num_samples=epoch_size, distribution=opts.data_distribution, task=task))
                training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
                model_common.train()
                set_decode_type(model_common, "sampling")
                loss = 0
                for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
                    x, bl_val = baseline.unwrap_batch(batch)
                    x = move_to(x, opts.device)
                    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
                    if eps > 0:
                        x = get_hard_samples(model_common, x, eps, batch_size=x.size(0), baseline=baseline)
                        if bl_val is not None:
                            bl_val, _ = baseline.eval(x, None)
                    model_common.train()
                    set_decode_type(model_common, "sampling")
                    cost, log_likelihood = model_common(x)
                    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
                    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
                    batch_loss = reinforce_loss + bl_loss
                    loss += batch_loss * x.size(0)
                loss = loss / epoch_size
                optimizer_common.zero_grad()
                loss.backward()
                clip_grad_norms(optimizer_common.param_groups, opts.max_grad_norm)
                optimizer_common.step()
                lr_scheduler.step()
                if epoch % opts.baseline_every_Xepochs_for_META == 0:
                    # avg_reward = validate(model_common, val_dataset, opts)
                    baseline.epoch_callback(model_common, epoch)

        if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            print('Saving model and state...')
            save_checkpoint(model_common, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

        # add validation here.
        pass


if __name__ == "__main__":
    run(get_options())
