import os
import copy
import math
import time
import random
import torch
from logging import getLogger
from collections import OrderedDict
from torch.optim import Adam as Optimizer

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from ProblemDef import get_random_problems, generate_task_set
from utils.utils import *
from utils.functions import *
from CVRP_baseline import *


class CVRPTrainer:
    """
    Implementation of POMO under the same training setting of POMO + meta-learning methods.
    """
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 meta_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.meta_params = meta_params
        assert self.meta_params['data_type'] == "size_distribution", "Not supported, need to modify the code!"

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model_params["norm"] = "batch"  # Original "POMO" Paper uses batch normalization
        self.model = Model(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.task_set = generate_task_set(self.meta_params)
        self.val_data, self.val_opt = {}, {}  # for lkh3_offline
        if self.meta_params["data_type"] == "size_distribution":
            # hardcoded - task_set: range(self.min_n, self.max_n, self.task_interval) * self.num_dist
            self.min_n, self.max_n, self.task_interval, self.num_dist = 50, 200, 5, 11
            self.task_w = torch.full(((self.max_n - self.min_n) // 5 + 1, self.num_dist), 1 / self.num_dist)

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        pretrain_load = trainer_params['pretrain_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Checkpoint loaded successfully from {}'.format(checkpoint_fullname))

        elif pretrain_load['enable']:  # meta-training on a pretrain model
            self.logger.info(">> Loading pretrained model: be careful with the type of the normalization layer!")
            checkpoint_fullname = '{path}'.format(**pretrain_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Pretrained model loaded successfully from {}'.format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        start_time = time.time()
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.meta_params['epochs']+1):
            self.logger.info('=================================================================')

            # lr decay (by 10) to speed up convergence at 90th iteration
            if epoch in [int(self.meta_params['epochs'] * 0.9)]:
                self.optimizer_params['optimizer']['lr'] /= 10
                for group in self.optimizer.param_groups:
                    group["lr"] /= 10
                    print(">> LR decay to {}".format(group["lr"]))

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']
            # Val
            no_aug_score_list = []
            if self.meta_params["data_type"] == "size_distribution":
                dir = "../../data/CVRP/Size_Distribution/"
                paths = ["cvrp200_uniform.pkl", "cvrp300_rotation.pkl"]
            if epoch <= 1 or (epoch % img_save_interval) == 0:
                for val_path in paths:
                    no_aug_score = self._fast_val(self.model, path=os.path.join(dir, val_path), val_episodes=64)
                    no_aug_score_list.append(round(no_aug_score, 4))
                self.result_log.append('val_score', epoch, no_aug_score_list)

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.meta_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}({:.2f}%): Time Est.: Elapsed[{}], Remain[{}], Val Score: {}".format(
                epoch, self.meta_params['epochs'], epoch / self.meta_params['epochs'] * 100, elapsed_time_str, remain_time_str, no_aug_score_list))

            all_done = (epoch == self.meta_params['epochs'])

            if epoch > 1 and (epoch % img_save_interval) == 0:  # save latest images, every X epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'], self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        """
        POMO Training, equivalent to the original POMO implementation.
        """
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        batch_size = self.meta_params['meta_batch_size']

        # Adaptive task scheduler - Not implemented for "size" and "distribution"
        if self.meta_params['curriculum']:
            if self.meta_params["data_type"] == "size_distribution":
                start = self.min_n + int(min(epoch / self.meta_params['sch_epoch'], 1) * (self.max_n - self.min_n))  # linear
                # start = self.min_n + int(1 / 2 * (1 - math.cos(math.pi * min(epoch / self.meta_params['sch_epoch'], 1))) * (self.max_n - self.min_n))  # cosine
                n = start // 5 * 5
                idx = (n - self.min_n) // 5
                tasks, weights = self.task_set[idx * 11: (idx + 1) * 11], self.task_w[idx]
                if epoch % self.meta_params['update_weight'] == 0:
                    self.task_w[idx] = self._update_task_weight(tasks, weights, epoch)

        # sample a batch of tasks
        for b in range(self.meta_params['B']):
            for step in range(self.meta_params['k']):
                if self.meta_params["data_type"] == "size_distribution":
                    task_params = tasks[torch.multinomial(self.task_w[idx], 1).item()] if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
                    batch_size = self.meta_params['meta_batch_size'] if task_params[0] <= 150 else self.meta_params['meta_batch_size'] // 2

                data = self._get_data(batch_size, task_params)
                env_params = {'problem_size': data[-1].size(1), 'pomo_size': data[-1].size(1)}
                avg_score, avg_loss = self._train_one_batch(data, Env(**env_params))
                score_AM.update(avg_score.item(), batch_size)
                loss_AM.update(avg_loss.item(), batch_size)

        # Log Once, for each epoch
        self.logger.info('Meta Iteration {:3d}: Score: {:.4f},  Loss: {:.4f}'.format(epoch, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, data, env):

        self.model.train()
        batch_size = data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # update model
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        print(score_mean)

        return score_mean, loss_mean

    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=32, return_all=False):
        aug_factor = 1
        if data is None:
            data = load_dataset(path)[offset: offset+val_episodes]  # load dataset from file
            depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
            depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity)
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)
        env = Env(**{'problem_size': data[-1].size(1), 'pomo_size': data[-1].size(1)})

        model.eval()
        batch_size = data[-1].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        print(no_aug_score)

        if return_all:
            return -max_pomo_reward[0, :].float()
        else:
            return no_aug_score.detach().item()

    def _get_data(self, batch_size, task_params, return_capacity=False):
        """
        Return CVRP data with the form of:
        depot_xy: [batch_size, 1, 2]
        node_xy: [batch_size, problem_size, 2]
        node_demand (unnormalized): [batch_size, problem_size]
        capacity: [batch_size]
        """
        if self.meta_params['data_type'] == 'distribution':
            assert len(task_params) == 2
            data = get_random_problems(batch_size, self.env_params['problem_size'], num_modes=task_params[0], cdist=task_params[1], distribution='gaussian_mixture', problem="cvrp")
        elif self.meta_params['data_type'] == 'size':
            assert len(task_params) == 1
            data = get_random_problems(batch_size, task_params[0], num_modes=0, cdist=0, distribution='uniform', problem="cvrp")
        elif self.meta_params['data_type'] == "size_distribution":
            assert len(task_params) == 3
            data = get_random_problems(batch_size, task_params[0], num_modes=task_params[1], cdist=task_params[2], distribution='gaussian_mixture', problem="cvrp")
        else:
            raise NotImplementedError

        # normalized node_demand by capacity & only return (depot_xy, node_xy, node_demand)
        if len(data) == 4 and not return_capacity:
            depot_xy, node_xy, node_demand, capacity = data
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)

        return data

    def _update_task_weight(self, tasks, weights, epoch):
        """
        Update the weights of tasks.
        For LKH3, set MAX_TRIALS = 100 to reduce time.
        """
        global run_func
        start_t, gap = time.time(), torch.zeros(weights.size(0))
        batch_size = 200 if self.meta_params["solver"] == "lkh3_offline" else 50
        idx = torch.randperm(batch_size)[:50]
        for i in range(gap.size(0)):
            selected = tasks[i]
            data = self._get_data(batch_size=batch_size, task_params=selected, return_capacity=True)

            # only use lkh3 at the first iteration of updating task weights
            if self.meta_params["solver"] == "lkh3_offline":
                if selected not in self.val_data.keys():
                    self.val_data[selected] = data  # (depot, loc, demand, capacity)
                    opts = argparse.ArgumentParser()
                    opts.cpus, opts.n, opts.progress_bar_mininterval = None, None, 0.1
                    dataset = [attr.cpu().tolist() for attr in data]
                    dataset = [(dataset[0][i][0], dataset[1][i], [int(d) for d in dataset[2][i]], int(dataset[3][i])) for i in range(data[0].size(0))]
                    executable = get_lkh_executable()
                    def run_func(args):
                        return solve_lkh_log(executable, *args, runs=1, disable_cache=True, MAX_TRIALS=100)  # otherwise it directly loads data from dir
                    results, _ = run_all_in_pool(run_func, "./LKH3_result", dataset, opts, use_multiprocessing=False)
                    self.val_opt[selected] = [j[0] for j in results]
                data = [attr[idx] for attr in self.val_data[selected]]
                data = (data[0], data[1], data[2] / data[3].view(-1, 1))

            model_score = self._fast_val(self.model, data=data, return_all=True)
            model_score = model_score.tolist()

            if self.meta_params["solver"] == "lkh3_offline":
                lkh_score = [self.val_opt[selected][j] for j in idx.tolist()]
                gap_list = [(model_score[j] - lkh_score[j]) / lkh_score[j] * 100 for j in range(len(lkh_score))]
                gap[i] = sum(gap_list) / len(gap_list)
            else:
                raise NotImplementedError
        print(">> Finish updating task weights within {}s".format(round(time.time() - start_t, 2)))

        temp = 1.0
        gap_temp = torch.Tensor([i / temp for i in gap.tolist()])
        print(gap, temp)
        print(">> Old task weights: {}".format(weights))
        weights = torch.softmax(gap_temp, dim=0)
        print(">> New task weights: {}".format(weights))

        return weights
