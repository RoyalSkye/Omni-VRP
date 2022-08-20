import os
import copy
import time
from tqdm import tqdm
import torch
import math
import pickle
from datetime import datetime

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, return_all_costs=False, return_pi=False):
    # Validate
    print('Validating...')
    if return_pi:
        cost, pi = rollout(model, dataset, opts, return_pi=True)
    else:
        cost = rollout(model, dataset, opts, return_pi=False)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    if return_all_costs and return_pi:
        return avg_cost, cost, pi
    if return_all_costs and not return_pi:
        return avg_cost, cost
    return avg_cost


def rollout(model, dataset, opts, return_pi=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(move_to(bat, opts.device), return_pi=True)
        return cost.data.cpu(), pi.cpu()

    if not return_pi:
        return torch.cat([
            eval_model_bat(bat)[0]
            for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
        ], 0)
    else:
        cost_array, pi_array = [], []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            cost_, pi_ = eval_model_bat(bat)
            cost_array.append(cost_)
            pi_array.append(pi_)
        return torch.cat(cost_array, 0), torch.cat(pi_array, 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def get_hard_samples(model, data, eps=5, batch_size=1024, baseline=None):
    from torch.autograd import Variable
    model.eval()
    set_decode_type(model, "greedy")

    def minmax(xy_):
        '''
        min max batch of graphs [b,n,2]
        '''
        xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
        return xy_

    def get_hard(model, data, eps):
        data.requires_grad_()
        cost, ll, pi = model(data, return_pi=True)
        if baseline is not None:
            with torch.no_grad():
                cost_b, _ = baseline.eval(data, None)  # only support for rollout now.
            # cost, ll = model(data)
            delta = torch.autograd.grad(eps*((cost/cost_b)*ll).mean(), data)[0]
        else:
            # As dividend is viewed as constant, it can be omitted in gradient calculation.
            delta = torch.autograd.grad(eps*(cost*ll).mean(), data)[0]
        ndata = data.detach() + delta
        ndata = minmax(ndata)
        ndata = Variable(ndata, requires_grad=False)
        return ndata

    # dataloader = DataLoader(data, batch_size=batch_size)
    # hard = torch.cat([get_hard(model, data, eps) for data in dataloader], dim=0)
    # return hard
    return get_hard(model, data, eps)


def tune_and_test(task, model_meta, baseline, epoch, test_dataset, problem, tb_logger, opts, fine_tuning_dataset=None, dict_results_task_sample_iter_wise=None):
    """
    test_dataset: Test dataset
    fine_tuning_dataset: dataset used for fine-tuning the model
    TODO: 1. why not fine_tuning_dataset = test_dataset?
    """

    print("task  ", task)
    sequence_updated_reward = []
    step = 0
    start_time = time.time()
    COUNTER_FINE_TUNE = 0

    training_dataset = baseline.wrap_dataset(fine_tuning_dataset)
    num_fine_tune_step_epochs = opts.test_num_step_epochs  # not 30; it depends upon (fine tuning dataset used)
    num_batch_size = 256 if task['graph_size'] < 150 else 128

    print("size of fine tuning dataset ", len(fine_tuning_dataset))
    print("num_batch_size ", num_batch_size)

    rand_sampler = torch.utils.data.RandomSampler(training_dataset, num_samples=len(training_dataset), replacement=True)
    training_dataloader = DataLoader(training_dataset, batch_size=num_batch_size, num_workers=1, sampler=rand_sampler)
    model_task = copy.deepcopy(model_meta)

    avg_reward, all_costs = validate(model_task, test_dataset, opts,  return_all_costs=True, return_pi=False)
    print(" >> AVG_COST {}, BEFORE TUNING on task {}".format(avg_reward, task))

    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE] = {}
    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = all_costs
    # if opts.rescale_for_testing is not None:  # only for scratch part since we didn't want to train again.
    #     dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost']*(task['rescale_for_testing']/3.0)
    # if COUNTER_FINE_TUNE % 50 == 0:
    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['avg_cost'] = avg_reward.item()
    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['current_time'] = datetime.now()
    print("COUNTER FINE TUNE {}, AVG COSTS {}".format(COUNTER_FINE_TUNE, avg_reward.item()))

    sequence_updated_reward.append(avg_reward)
    model_task.train()
    set_decode_type(model_task, "sampling")
    optimizer = optim.Adam(model_task.parameters(), lr=opts.lr_model*0.1)
    print("num_fine_tune_step_epochs ", num_fine_tune_step_epochs)
    time_spent_in_fine_tuning = 0

    for outer_step_id in range(num_fine_tune_step_epochs):
        print("Fine-tune epoch ", outer_step_id)
        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
            # if time_spent_in_fine_tuning > 180 or (COUNTER_FINE_TUNE == 250000 and opts.longer_fine_tune == 0):
            # if COUNTER_FINE_TUNE > num_fine_tune_step_epochs:
            #     return updated_reward
            time_before_update = datetime.now()
            model_task.train()
            set_decode_type(model_task, "sampling")
            train_batch(model_task, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
            time_after_update = datetime.now()
            time_taken_for_update = (time_after_update - time_before_update).total_seconds() / 60.0
            time_spent_in_fine_tuning += time_taken_for_update
            step += 1
            COUNTER_FINE_TUNE += 1
            print(">> Time spent in fine-tuning {} minutes, {} steps".format(time_spent_in_fine_tuning, COUNTER_FINE_TUNE))

        # if COUNTER_FINE_TUNE % 10 == 0 or COUNTER_FINE_TUNE == 1:
        updated_reward, updated_all_costs = validate(model_task, test_dataset, opts, return_all_costs=True, return_pi=False)
        print(" COST AFTER TUNING ", updated_reward)
        sequence_updated_reward.append(updated_reward)
        # if dict_results_task_sample_iter_wise is not None:
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE] = {}
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = updated_all_costs
        # dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['pi'] = None
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['current_time'] = datetime.now()
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['time_spent_in_fine_tuning'] = time_spent_in_fine_tuning
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['avg_cost'] = updated_reward.item()

    epoch_duration = time.time() - start_time
    updated_reward = validate(model_task, test_dataset, opts, return_all_costs=False, return_pi=False)

    if num_fine_tune_step_epochs == 0:
        print("****** No fine tuning done **** ")
    else:
        print(">> {} steps within {} epochs fine-tuning finished, took {} s".format(COUNTER_FINE_TUNE, num_fine_tune_step_epochs, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
        print(">> AFTER TUNING on task ", task)
        print(">> COST AFTER TUNING ", updated_reward)

    for index, x in enumerate(sequence_updated_reward):
        print(x.item(), end=' -> ')
    print("")

    return updated_reward


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, task, eps=0):
    """
        Implementation of ordinary AM training (update by batch).
    """
    print("Start train epoch {}, lr={} on task {}".format(epoch, optimizer.param_groups[0]['lr'], task))
    step = 0
    start_time = time.time()

    # Generate new training data for each epoch
    epoch_size = opts.batch_size * opts.k_tune_steps
    training_dataset = baseline.wrap_dataset(problem.make_dataset(num_samples=epoch_size, distribution=opts.data_distribution, task=task))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts, eps=eps)
        step += 1

    # update baseline model
    if epoch % opts.baseline_every_Xepochs_for_META == 0:
        # avg_reward = validate(model, val_dataset, opts)
        baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))


def meta_train_epoch(model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, alpha, task, eps=0):
    """
        Implementation for meta-learning framework.
    """
    lr = opts.lr_model * (opts.lr_decay ** epoch)
    print("Start train epoch {}, lr={}, alpha={} on task {}".format(epoch, lr, alpha, task))
    step = 0
    start_time = time.time()

    # Generate new training data for each epoch
    epoch_size = opts.batch_size * opts.k_tune_steps
    training_dataset = baseline.wrap_dataset(problem.make_dataset(num_samples=epoch_size, distribution=opts.data_distribution, task=task))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    current_weights = copy.deepcopy(model_meta.state_dict())
    model_meta.train()
    set_decode_type(model_meta, "sampling")
    optimizer = optim.Adam(model_meta.parameters(), lr=lr)

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(model_meta, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts, eps=eps)
        step += 1

    candidate_weights = model_meta.state_dict()
    state_dict = {candidate: (current_weights[candidate] + alpha * (candidate_weights[candidate] - current_weights[candidate])) for candidate in candidate_weights}

    # update baseline model
    if epoch % opts.baseline_every_Xepochs_for_META == 0:
        # avg_reward = validate(model, val_dataset, opts)
        baseline.epoch_callback(model_meta, epoch)

    model_meta.load_state_dict(state_dict)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts, eps=0):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    if eps > 0:
        x = get_hard_samples(model, x, eps, batch_size=x.size(0), baseline=baseline)
        if bl_val is not None:
            bl_val, _ = baseline.eval(x, None)

    # Evaluate model, get costs and log probabilities
    model.train()
    set_decode_type(model, "sampling")
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    # if step % int(opts.log_step) == 0:
    #     log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
