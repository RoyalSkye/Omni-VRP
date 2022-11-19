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
    Implementation of POMO with MAML / FOMAML / Reptile on CVRP.
    For MAML & FOMAML, ref to "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks";
    For Reptile, ref to "On First-Order Meta-Learning Algorithms".
    Refer to "https://lilianweng.github.io/posts/2018-11-30-meta-learning"
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
        self.model_params["norm"] = None  # Original "POMO" Paper uses instance/batch normalization
        self.meta_model = Model(**self.model_params)
        self.meta_optimizer = Optimizer(self.meta_model.parameters(), **self.optimizer_params['optimizer'])
        self.alpha = self.meta_params['alpha']  # for reptile
        self.task_set = generate_task_set(self.meta_params)
        self.val_data, self.val_opt = {}, {}  # for lkh3_offline
        if self.meta_params["data_type"] == "size":
            self.min_n, self.max_n, self.task_interval = self.task_set[0][0], self.task_set[-1][0], 5  # [20, 150]
            self.task_w = {start: 1 / (len(self.task_set) // 5) for start in range(self.min_n, self.max_n, self.task_interval)}
            # self.task_w = torch.full((len(self.task_set)//self.task_interval,), 1/(len(self.task_set)//self.task_interval))
        elif self.meta_params["data_type"] == "distribution":
            self.task_w = torch.full((len(self.task_set),), 1 / len(self.task_set))
        else:
            raise NotImplementedError

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(">> Model loaded from {}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        start_time, best_mean = time.time(), 1000
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.meta_params['epochs'] + 1):
            self.logger.info('=================================================================')

            # lr decay (by 10) to speed up convergence at 90th and 95th iterations
            if epoch in [int(self.meta_params['epochs'] * 0.9), int(self.meta_params['epochs'] * 0.95)]:
                self.optimizer_params['optimizer']['lr'] /= 10
                for group in self.meta_optimizer.param_groups:
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
            if self.meta_params["data_type"] == "size":
                dir = "../../data/CVRP/Size/"
                paths = ["cvrp100_uniform.pkl", "cvrp200_uniform.pkl", "cvrp300_uniform.pkl"]
            elif self.meta_params["data_type"] == "distribution":
                dir = "../../data/CVRP/Distribution/"
                paths = ["cvrp100_uniform.pkl", "cvrp100_gaussian.pkl", "cvrp100_cluster.pkl", "cvrp100_diagonal.pkl", "cvrp100_cvrplib.pkl"]
            elif self.meta_params["data_type"] == "size_distribution":
                pass
            if epoch <= 1 or (epoch % img_save_interval) == 0:
                for val_path in paths:
                    no_aug_score = self._fast_val(self.meta_model, path=os.path.join(dir, val_path), val_episodes=64, mode="eval")
                    no_aug_score_list.append(round(no_aug_score, 4))
                self.result_log.append('val_score', epoch, no_aug_score_list)
                cur_mean = sum(no_aug_score_list) / len(no_aug_score_list)
                # save best checkpoint (conditioned on the val datasets!)
                if cur_mean < best_mean:
                    best_mean = cur_mean
                    self.best_meta_model = copy.deepcopy(self.meta_model)
                    self.logger.info("Saving (best) trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.meta_model.state_dict(),
                        'optimizer_state_dict': self.meta_optimizer.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, '{}/best_checkpoint.pt'.format(self.result_folder))

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
                    'model_state_dict': self.meta_model.state_dict(),
                    'optimizer_state_dict': self.meta_optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        """
        1. Sample B training tasks from task distribution P(T)
        2. Inner-loop: for a batch of tasks T_i, POMO training -> \theta_i
        3. Outer-loop: update meta-model -> \theta_0
        """
        self.meta_optimizer.zero_grad()
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        batch_size = self.meta_params['meta_batch_size']

        """
        Adaptive task scheduler:
            for size: gradually increase the problem size (Curriculum learning);
            for distribution: we compute the relative gaps (w.r.t. LKH3) or estimate the potential improvements of each distribution (i.e., bootstrap) every X iters;
        """
        start, end = 0, 0
        pass

        self._alpha_scheduler(epoch)  # for reptile
        fast_weights, val_loss, meta_grad_dict = [], 0, {(i, j): 0 for i, group in enumerate(self.meta_optimizer.param_groups) for j, _ in enumerate(group['params'])}

        for b in range(self.meta_params['B']):
            # sample a task
            if self.meta_params["data_type"] == "size":
                task_params = random.sample(range(start, end + 1), 1) if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
                # batch_size = self.meta_params['meta_batch_size'] if task_params[0] <= 100 else self.meta_params['meta_batch_size'] // 2
            elif self.meta_params["data_type"] == "distribution":
                task_params = self.task_set[torch.multinomial(self.task_w, 1).item()] if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
            elif self.meta_params["data_type"] == "size_distribution":
                pass

            # preparation
            if self.meta_params['meta_method'] in ['fomaml', 'reptile']:
                task_model = copy.deepcopy(self.meta_model)
                optimizer = Optimizer(task_model.parameters(), **self.optimizer_params['optimizer'])
                optimizer.load_state_dict(self.meta_optimizer.state_dict())
            elif self.meta_params['meta_method'] == 'maml':
                if self.model_params['meta_update_encoder']:
                    fast_weight = OrderedDict(self.meta_model.named_parameters())
                else:
                    fast_weight = OrderedDict(self.meta_model.decoder.named_parameters())
                    for k in list(fast_weight.keys()):
                        fast_weight["decoder." + k] = fast_weight.pop(k)

            # inner-loop optimization
            for step in range(self.meta_params['k']):
                data = self._get_data(batch_size, task_params)
                env_params = {'problem_size': data[-1].size(1), 'pomo_size': data[-1].size(1)}
                self.meta_model.train()
                if self.meta_params['meta_method'] in ['reptile', 'fomaml']:
                    avg_score, avg_loss = self._train_one_batch(task_model, data, Env(**env_params), optimizer)
                elif self.meta_params['meta_method'] == 'maml':
                    avg_score, avg_loss, fast_weight = self._train_one_batch_maml(fast_weight, data, Env(**env_params))
                score_AM.update(avg_score.item(), batch_size)
                loss_AM.update(avg_loss.item(), batch_size)

            val_data = self._get_val_data(batch_size, task_params)
            self.meta_model.train()
            if self.meta_params['meta_method'] == 'maml':
                val_loss = self._fast_val(fast_weight, data=val_data, mode="maml") / self.meta_params['B']
                self.meta_optimizer.zero_grad()
                val_loss.backward()
                for i, group in enumerate(self.meta_optimizer.param_groups):
                    for j, p in enumerate(group['params']):
                        meta_grad_dict[(i, j)] += p.grad
            elif self.meta_params['meta_method'] == 'fomaml':
                val_loss = self._fast_val(task_model, data=val_data, mode="fomaml") / self.meta_params['B']
                optimizer.zero_grad()
                val_loss.backward()
                for i, group in enumerate(optimizer.param_groups):
                    for j, p in enumerate(group['params']):
                        meta_grad_dict[(i, j)] += p.grad
            elif self.meta_params['meta_method'] == 'reptile':
                fast_weights.append(task_model.state_dict())

        # outer-loop optimization (update meta-model)
        if self.meta_params['meta_method'] == 'maml':
            self.meta_optimizer.zero_grad()
            for i, group in enumerate(self.meta_optimizer.param_groups):
                for j, p in enumerate(group['params']):
                    p.grad = meta_grad_dict[(i, j)]
            self.meta_optimizer.step()
        elif self.meta_params['meta_method'] == 'fomaml':
            self.meta_optimizer.zero_grad()
            for i, group in enumerate(self.meta_optimizer.param_groups):
                for j, p in enumerate(group['params']):
                    p.grad = meta_grad_dict[(i, j)]
            self.meta_optimizer.step()
        elif self.meta_params['meta_method'] == 'reptile':
            state_dict = {params_key: (self.meta_model.state_dict()[params_key] + self.alpha * torch.mean(torch.stack([fast_weight[params_key] - self.meta_model.state_dict()[params_key] for fast_weight in fast_weights], dim=0), dim=0)) for params_key in self.meta_model.state_dict()}
            self.meta_model.load_state_dict(state_dict)

        # Log Once, for each epoch
        self.logger.info('Meta Iteration {:3d}: alpha: {:6f}, Score: {:.4f},  Loss: {:.4f}'.format(epoch, self.alpha, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, task_model, data, env, optimizer=None):

        task_model.train()
        batch_size = data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        task_model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = task_model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # update model
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        print(score_mean)

        return score_mean, loss_mean

    def _train_one_batch_maml(self, fast_weight, data, env, optimizer=None):

        batch_size = data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.meta_model.pre_forward(reset_state, weights=fast_weight)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = self.meta_model(state, weights=fast_weight)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # 1. update model - in SGD way
        # gradients = torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=True)  # allow_unused=True
        # fast_weight = OrderedDict(
        #     (name, param - self.optimizer_params['optimizer']['lr'] * grad)
        #     for ((name, param), grad) in zip(fast_weight.items(), gradients)
        # )
        # 2. update model - in Adam way
        gradients = torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=True)  # allow_unused=True
        w_t, (beta1, beta2), eps = [], self.meta_optimizer.param_groups[0]['betas'], self.meta_optimizer.param_groups[0]['eps']
        lr, weight_decay = self.optimizer_params['optimizer']['lr'], self.optimizer_params['optimizer']['weight_decay']
        for i, ((name, param), grad) in enumerate(zip(fast_weight.items(), gradients)):
            if self.meta_optimizer.state_dict()['state'] != {}:
                i = i if self.model_params['meta_update_encoder'] else i + 58  # i \in [0, 62]
                state = self.meta_optimizer.state_dict()['state'][i]
                step, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']
                step += 1
                step = step.item()
                # compute grad based on Adam source code using in-place operation
                # update Adam stat (step, exp_avg and exp_avg_sq have already been updated by in-place operation)
                # may encounter RuntimeError: (a leaf Variable that requires grad) / (the tensor used during grad computation) cannot use in-place operation.
                grad = grad.add(param, alpha=weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                # param.addcdiv_(exp_avg, denom, value=-step_size)
                param = param - step_size * exp_avg / denom
                self.meta_optimizer.state_dict()['state'][i]['exp_avg'] = exp_avg.clone().detach()
                self.meta_optimizer.state_dict()['state'][i]['exp_avg_sq'] = exp_avg_sq.clone().detach()
            else:
                param = param - lr * grad
            w_t.append((name, param))
        fast_weight = OrderedDict(w_t)

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        print(score_mean)

        return score_mean, loss_mean, fast_weight

    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=32, mode="eval", return_all=False):
        aug_factor = 1
        if data is None:
            data = load_dataset(path)[offset: offset + val_episodes]  # load dataset from file
            depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
            depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity)
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)
        env = Env(**{'problem_size': data[-1].size(1), 'pomo_size': data[-1].size(1)})

        batch_size = data[-1].size(0)
        if mode == "eval":
            model.eval()
            with torch.no_grad():
                env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
                reset_state, _, _ = env.reset()
                model.pre_forward(reset_state)
                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = model(state)
                    # shape: (batch, pomo)
                    state, reward, done = env.step(selected)

        elif mode in ["maml", "fomaml"]:
            fast_weight = model
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            if mode == "maml":
                self.meta_model.pre_forward(reset_state, weights=fast_weight)
            else:
                model.pre_forward(reset_state)
            prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
            state, reward, done = env.pre_step()
            while not done:
                if mode == "maml":
                    selected, prob = self.meta_model(state, weights=fast_weight)
                else:
                    selected, prob = model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
            loss_mean = loss.mean()
        else:
            raise NotImplementedError

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        print(no_aug_score)

        if mode == "eval":
            if return_all:
                return -max_pomo_reward[0, :].float()
            else:
                return no_aug_score.detach().item()
        else:
            return loss_mean

    def _bootstrap(self, fast_weight, data, mode="eval"):
        """
        mode = "maml": Ref to "Bootstrap Meta-Learning", ICLR 2022 (not implemented for CVRP);
        mode = "eval": Used to update task weights.
        """
        assert mode in ["eval"], "{} not implemented!".format(mode)
        bootstrap_weight = fast_weight
        batch_size, aug_factor = data[-1].size(0), 1
        bootstrap_reward = torch.full((batch_size, 1), float("-inf"))
        optimizer = Optimizer(bootstrap_weight.parameters(), **self.optimizer_params['optimizer'])
        # optimizer.load_state_dict(self.meta_optimizer.state_dict())
        with torch.enable_grad():
            for L in range(self.meta_params['bootstrap_steps']):
                env = Env(**{'problem_size': data[-1].size(1), 'pomo_size': data[-1].size(1)})
                env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
                reset_state, _, _ = env.reset()
                bootstrap_weight.pre_forward(reset_state)
                prob_list = torch.zeros(size=(aug_factor * batch_size, env.pomo_size, 0))
                state, reward, done = env.pre_step()
                while not done:
                    selected, prob = bootstrap_weight(state)
                    state, reward, done = env.step(selected)  # (aug_factor * batch_size, pomo_size)
                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

                # (batch, augmentation * pomo)
                reward = reward.reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
                advantage = reward - reward.float().mean(dim=1, keepdims=True)
                log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
                loss = -advantage * log_prob
                loss_mean = loss.mean()

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

                max_pomo_reward, _ = reward.max(dim=1)
                max_pomo_reward = max_pomo_reward.view(-1, 1)
                bootstrap_reward = torch.where(max_pomo_reward > bootstrap_reward, max_pomo_reward, bootstrap_reward)  # (batch_size, 1)

        return bootstrap_reward

    def _get_data(self, batch_size, task_params):
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
        if len(data) == 4:
            depot_xy, node_xy, node_demand, capacity = data
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)

        return data

    def _get_val_data(self, batch_size, task_params):
        if self.meta_params["data_type"] == "size":
            start1, end1 = min(task_params[0] + 10, self.max_n), min(task_params[0] + 20, self.max_n)
            val_size = random.sample(range(start1, end1 + 1), 1)[0]
            val_data = self._get_data(batch_size, (val_size,))
            # val_data = self._get_data(batch_size, task_params)  # TODO: which is better?
        elif self.meta_params["data_type"] == "distribution":
            val_data = self._get_data(batch_size, task_params)
        elif self.meta_params["data_type"] == "size_distribution":
            pass
        else:
            raise NotImplementedError

        return val_data

    def _alpha_scheduler(self, epoch):
        """
        Update param for Reptile.
        """
        self.alpha = max(self.alpha * self.meta_params['alpha_decay'], 0.0001)

    def _update_task_weight(self, epoch):
        pass
