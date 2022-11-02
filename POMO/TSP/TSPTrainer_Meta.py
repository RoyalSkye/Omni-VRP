import os
import copy
import math
import time
import random
import torch
from logging import getLogger
from collections import OrderedDict

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
# from torch.optim import SGD as Optimizer
from TSProblemDef import get_random_problems, generate_task_set

from utils.utils import *
from utils.functions import *
from TSP_baseline import *


class TSPTrainer:
    """
    Implementation of POMO with MAML / FOMAML / Reptile on TSP.
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
        self.meta_model = Model(**self.model_params)
        self.meta_optimizer = Optimizer(self.meta_model.parameters(), **self.optimizer_params['optimizer'])
        self.alpha = self.meta_params['alpha']  # for reptile
        self.task_set = generate_task_set(self.meta_params)
        if self.meta_params["data_type"] == "size":
            self.min_n, self.max_n, self.task_interval = self.task_set[0][0], self.task_set[-1][0], 5  # [20, 150] / [0, 100]
            # self.task_w = {start: 1/(len(self.task_set)//5) for start in range(self.min_n, self.max_n, self.task_interval)}
            self.task_w = torch.full((len(self.task_set)//self.task_interval,), 1/(len(self.task_set)//self.task_interval))
            # self.ema_est = {i[0]: 1 for i in self.task_set}
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
        for epoch in range(self.start_epoch, self.meta_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']
            # Val
            dir, no_aug_score_list = "../../data/TSP/", []
            if self.meta_params["data_type"] == "size":
                paths = ["tsp50_uniform.pkl", "tsp100_uniform.pkl", "tsp200_uniform.pkl"]
            elif self.meta_params["data_type"] == "distribution":
                paths = ["tsp100_uniform.pkl", "tsp100_gaussian.pkl", "tsp100_cluster.pkl", "tsp100_diagonal.pkl", "tsp100_tsplib.pkl"]
            elif self.meta_params["data_type"] == "size_distribution":
                pass
            if epoch <= 1 or (epoch % img_save_interval) == 0:
                for val_path in paths:
                    no_aug_score = self._fast_val(self.meta_model, path=os.path.join(dir, val_path), val_episodes=64, mode="eval")
                    no_aug_score_list.append(round(no_aug_score, 4))
                self.result_log.append('val_score', epoch, no_aug_score_list)
                cur_mean = sum(no_aug_score_list) / len(no_aug_score_list)
                # save best checkpoint
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
                epoch, self.meta_params['epochs'], epoch/self.meta_params['epochs']*100, elapsed_time_str, remain_time_str, no_aug_score_list))

            if self.trainer_params['stop_criterion'] == "epochs":
                all_done = (epoch == self.meta_params['epochs'])
            else:
                all_done = (time.time() - start_time) >= self.trainer_params['time_limit']

            if epoch > 1 and (epoch % img_save_interval) == 0:  # save latest images, every X epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'], self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                # save checkpoint
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.meta_model.state_dict(),
                    'optimizer_state_dict': self.meta_optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # if all_done or (epoch % img_save_interval) == 0:
            #     image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['train_score'])
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_score'])
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'], self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        """
        1. Sample B training tasks from task distribution P(T)
        2. inner-loop: for a batch of tasks T_i, do reptile -> \theta_i
        3. outer-loop: update meta-model -> \theta_0
        """
        self.meta_optimizer.zero_grad()
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        """
        Curriculum learning / Adaptive task scheduler:
            for size: gradually increase the problem size
            for distribution: adversarial budgets (i.e., \epsilon) may not be correlated with the hardness of constructed 
                              data distribution. Instead, we evaluate the relative gaps (w.r.t. LKH3) of dist/eps sampled 
                              from each interval every X iters. Hopefully, it can indicate the hardness of its neighbor.
        """
        if self.meta_params["data_type"] == "size":
            # start = self.min_n + int(epoch/self.meta_params['epochs'] * (self.max_n - self.min_n))  # linear
            start = self.min_n + int(1/2 * (1-math.cos(math.pi * min(epoch/self.meta_params['epochs'], 1))) * (self.max_n - self.min_n))  # cosine
            end = min(start + 10, self.max_n)  # 10 is the size of the sliding window
            if self.meta_params["curriculum"]: print(">> training task {}".format((start, end)))
        elif self.meta_params["data_type"] == "distribution":
            if epoch != 0 and epoch % self.meta_params['update_weight'] == 0:
                self._update_task_weight(epoch)
            # start = int(min(epoch/self.meta_params["epochs"], 1) * (len(self.task_w)-1))  # linear
            # start = int(1/2 * (1-math.cos(math.pi * min(epoch/self.meta_params['epochs'], 1))) * (len(self.task_w)-1))  # cosine
            # end = min(start + 5, len(self.task_w))
        elif self.meta_params["data_type"] == "size_distribution":
            pass

        self._alpha_scheduler(epoch)
        fast_weights, val_loss, fomaml_grad = [], 0, []

        # sample a batch of tasks
        for i in range(self.meta_params['B']):
            if self.meta_params["data_type"] == "size":
                task_params = random.sample(range(start, end+1), 1) if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
                batch_size = self.meta_params['meta_batch_size'] if task_params[0] <= 100 else self.meta_params['meta_batch_size'] // 2
            elif self.meta_params["data_type"] == "distribution":
                # sample based on task weights
                task_params = self.task_set[torch.multinomial(self.task_w, 1).item()] if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
                # task_params = self.task_set[random.sample(torch.topk(self.task_w, 10)[1].tolist(), 1)[0]] if self.meta_params['curriculum'] else random.sample(self.task_set, 1)[0]
                # curri: from easy task (small gaps) -> hard task (large gaps)
                # selected_idx = torch.sort(self.task_w, descending=False)[1].tolist()[start: end]
                # task_params = self.task_set[random.sample(selected_idx, 1)[0]] if self.meta_params['curriculum'] and epoch >= self.meta_params['update_weight'] else random.sample(self.task_set, 1)[0]
                batch_size = self.meta_params['meta_batch_size']
            elif self.meta_params["data_type"] == "size_distribution":
                pass

            if self.meta_params['meta_method'] in ['fomaml', 'reptile']:
                task_model = copy.deepcopy(self.meta_model)
                optimizer = Optimizer(task_model.parameters(), **self.optimizer_params['optimizer'])
            elif self.meta_params['meta_method'] == 'maml':
                if self.model_params['meta_update_encoder']:
                    fast_weight = OrderedDict(self.meta_model.named_parameters())
                else:
                    fast_weight = OrderedDict(self.meta_model.decoder.named_parameters())
                    for k in list(fast_weight.keys()):
                        fast_weight["decoder."+k] = fast_weight.pop(k)
                optimizer = Optimizer(fast_weight.values(), **self.optimizer_params['optimizer'])
            optimizer.load_state_dict(self.meta_optimizer.state_dict())

            # inner-loop optimization
            for step in range(self.meta_params['k']):
                data = self._get_data(batch_size, task_params)
                env_params = {'problem_size': data.size(1), 'pomo_size': data.size(1)}
                self.meta_model.train()
                if self.meta_params['meta_method'] in ['reptile', 'fomaml']:
                    avg_score, avg_loss = self._train_one_batch(task_model, data, Env(**env_params), optimizer)
                elif self.meta_params['meta_method'] == 'maml':
                    avg_score, avg_loss, fast_weight = self._train_one_batch_maml(fast_weight, data, Env(**env_params), optimizer)
                score_AM.update(avg_score.item(), batch_size)
                loss_AM.update(avg_loss.item(), batch_size)

            val_data = self._get_val_data(batch_size, task_params)
            self.meta_model.train()
            if self.meta_params['meta_method'] == 'maml':
                val_loss = self._fast_val(fast_weight, data=val_data, mode="maml")
                val_loss /= self.meta_params['B']
                val_loss.backward()
            elif self.meta_params['meta_method'] == 'fomaml':
                val_loss = self._fast_val(task_model, data=val_data, mode="fomaml")
                grad = torch.autograd.grad(val_loss, task_model.parameters())
                fomaml_grad.append(grad)
                self.meta_optimizer.load_state_dict(optimizer.state_dict())
            elif self.meta_params['meta_method'] == 'reptile':
                fast_weights.append(task_model.state_dict())

        # outer-loop optimization (update meta-model)
        if self.meta_params['meta_method'] == 'maml':
            # val_loss /= self.meta_params['B']
            # self.meta_optimizer.zero_grad()
            # val_loss.backward()
            # print(self.meta_model.encoder.embedding.weight.grad.norm(p=2).cpu().item())
            # print(self.meta_model.decoder.multi_head_combine.weight.grad.norm(p=2).cpu().item())
            # grad_norms = clip_grad_norms(self.meta_optimizer.param_groups, max_norm=1.0)
            # print(grad_norms[0])
            self.meta_optimizer.step()
        elif self.meta_params['meta_method'] == 'fomaml':
            updated_weights = self.meta_model.state_dict()
            for gradients in fomaml_grad:
                updated_weights = OrderedDict(
                    (name, param - self.optimizer_params['optimizer']['lr'] / self.meta_params['B'] * grad)
                    for ((name, param), grad) in zip(updated_weights.items(), gradients)
                )
            self.meta_model.load_state_dict(updated_weights)
        elif self.meta_params['meta_method'] == 'reptile':
            state_dict = {params_key: (self.meta_model.state_dict()[params_key] + self.alpha * torch.mean(torch.stack([fast_weight[params_key] - self.meta_model.state_dict()[params_key] for fast_weight in fast_weights], dim=0), dim=0)) for params_key in self.meta_model.state_dict()}
            self.meta_model.load_state_dict(state_dict)

        # Log Once, for each epoch
        self.logger.info('Meta Iteration {:3d}: alpha: {:6f}, Score: {:.4f},  Loss: {:.4f}'.format(epoch, self.alpha, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, task_model, data, env, optimizer=None):

        task_model.train()
        batch_size = data.size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        task_model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout, please note that the reward is negative (i.e., -length of route).
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = task_model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss & adjust reward
        # self.ema_est[data.size(1)] = 0.99 * self.ema_est[data.size(1)] + (1 - 0.99) * (-reward.float().mean().item()) if self.ema_est[data.size(1)] != 1 else -reward.float().mean().item()
        # reward = reward / self.ema_est[data.size(1)]
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

        batch_size = data.size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.meta_model.pre_forward(reset_state, weights=fast_weight)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout, please note that the reward is negative (i.e., -length of route).
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = self.meta_model(state, weights=fast_weight)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss & adjust reward
        # self.ema_est[data.size(1)] = 0.99 * self.ema_est[data.size(1)] + (1 - 0.99) * (-reward.float().mean().item()) if self.ema_est[data.size(1)] != 1 else -reward.float().mean().item()
        # print(self.ema_est)
        # reward = reward / self.ema_est[data.size(1)]
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # update model
        # gradients = torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=True)  # allow_unused=True
        # fast_weight = OrderedDict(
        #     (name, param - self.optimizer_params['optimizer']['lr'] * grad)
        #     for ((name, param), grad) in zip(fast_weight.items(), gradients)
        # )
        optimizer.zero_grad()
        # torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=True)
        loss_mean.backward(retain_graph=True, create_graph=True)
        optimizer.step()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        print(score_mean)

        return score_mean, loss_mean, fast_weight

    def _fast_val(self, model, data=None, path=None, val_episodes=32, mode="eval", return_all=False):
        aug_factor = 1
        data = torch.Tensor(load_dataset(path)[: val_episodes]) if data is None else data
        env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})

        batch_size = data.size(0)
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

            """
            if self.meta_params['bootstrap_steps'] != 0:
                bootstrap_reward = self._bootstrap(fast_weight, data, mode="maml")
                advantage = reward - bootstrap_reward
            else:
                # self.ema_est[data.size(1)] = 0.99 * self.ema_est[data.size(1)] + (1 - 0.99) * (-reward.float().mean().item()) if self.ema_est[data.size(1)] != 1 else -reward.float().mean().item()
                # reward = reward / self.ema_est[data.size(1)]
                advantage = reward - reward.float().mean(dim=1, keepdims=True)
            """
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
        mode = "maml": Ref to "Bootstrap Meta-Learning", ICLR 2022;
        mode = "eval": Used to update task weights.
        """
        bootstrap_weight = fast_weight
        batch_size, aug_factor = data.size(0), 8
        bootstrap_reward = torch.full((batch_size, 1), float("-inf"))
        if mode == "eval":
            optimizer = Optimizer(bootstrap_weight.parameters(), **self.optimizer_params['optimizer'])
            # optimizer.load_state_dict(self.meta_optimizer.state_dict())
        with torch.enable_grad():
            for L in range(self.meta_params['bootstrap_steps']):
                env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})
                env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
                reset_state, _, _ = env.reset()
                if mode == "maml":
                    self.meta_model.pre_forward(reset_state, weights=bootstrap_weight)
                elif mode == "eval":
                    bootstrap_weight.pre_forward(reset_state)
                prob_list = torch.zeros(size=(aug_factor * batch_size, env.pomo_size, 0))
                state, reward, done = env.pre_step()
                while not done:
                    if mode == "maml":
                        selected, prob = self.meta_model(state, weights=bootstrap_weight)
                    elif mode == "eval":
                        selected, prob = bootstrap_weight(state)
                    state, reward, done = env.step(selected)  # (aug_factor * batch_size, pomo_size)
                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

                # (batch, augmentation * pomo)
                reward = reward.reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
                advantage = reward - reward.float().mean(dim=1, keepdims=True)
                log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
                loss = -advantage * log_prob
                loss_mean = loss.mean()

                if mode == "maml":
                    # TODO: need to update
                    gradients = torch.autograd.grad(loss_mean, bootstrap_weight.values(), create_graph=False)
                    bootstrap_weight = OrderedDict(
                        (name, param - self.optimizer_params['optimizer']['lr'] * grad)
                        for ((name, param), grad) in zip(bootstrap_weight.items(), gradients)
                    )
                elif mode == "eval":
                    optimizer.zero_grad()
                    loss_mean.backward()
                    optimizer.step()

                max_pomo_reward, _ = reward.max(dim=1)
                max_pomo_reward = max_pomo_reward.view(-1, 1)
                bootstrap_reward = torch.where(max_pomo_reward > bootstrap_reward, max_pomo_reward, bootstrap_reward)  # (batch_size, 1)

        return bootstrap_reward


    def _get_data(self, batch_size, task_params):

        if self.meta_params['data_type'] == 'distribution':
            assert len(task_params) == 2
            data = get_random_problems(batch_size, self.env_params['problem_size'], num_modes=task_params[0], cdist=task_params[1], distribution='gaussian_mixture')
        elif self.meta_params['data_type'] == 'size':
            assert len(task_params) == 1
            data = get_random_problems(batch_size, task_params[0], num_modes=0, cdist=0, distribution='uniform')
        elif self.meta_params['data_type'] == "size_distribution":
            assert len(task_params) == 3
            data = get_random_problems(batch_size, task_params[0], num_modes=task_params[1], cdist=task_params[2], distribution='gaussian_mixture')
        else:
            raise NotImplementedError

        return data

    def _get_val_data(self, batch_size, task_params):
        if self.meta_params["data_type"] == "size":
            start1, end1 = min(task_params[0] + 10, self.max_n), min(task_params[0] + 20, self.max_n)
            val_size = random.sample(range(start1, end1 + 1), 1)[0]
            val_data = self._get_data(batch_size, (val_size,))
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

    def _generate_x_adv(self, data, eps=10.0):
        """
        Generate adversarial data based on the current model, also need to generate optimal sol for x_adv.
        """
        from torch.autograd import Variable
        def minmax(xy_):
            # min_max normalization: [b,n,2]
            xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
            return xy_

        if eps == 0: return data
        # generate x_adv
        self.meta_model.eval()
        aug_factor, batch_size = 1, data.size(0)
        env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})
        with torch.enable_grad():
            data.requires_grad_()
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            self.meta_model.pre_forward(reset_state)
            prob_list = torch.zeros(size=(aug_factor * batch_size, env.pomo_size, 0))
            state, reward, done = env.pre_step()
            while not done:
                selected, prob = self.meta_model(state)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).view(batch_size, -1)
            baseline_reward = aug_reward.float().mean(dim=1, keepdims=True)
            advantage = aug_reward - baseline_reward
            log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).view(batch_size, -1)

            # delta = torch.autograd.grad(eps * ((advantage / baseline_reward) * log_prob).mean(), data)[0]
            delta = torch.autograd.grad(eps * ((-advantage) * log_prob).mean(), data)[0]
            data = data.detach() + delta
            data = minmax(data)
            data = Variable(data, requires_grad=False)

        # generate opt sol
        # opt_sol = solve_all_gurobi(data)
        # return data, opt_sol

        return data

    def _update_task_weight(self, epoch):
        """
        Update the weights of tasks.
        """
        start_t, gap = time.time(), torch.zeros(self.task_w.size(0))
        for i in range(gap.size(0)):
            if self.meta_params["data_type"] == "size":
                start = i * self.task_interval
                end = min(start + self.task_interval, self.max_n)
                selected = random.sample([j for j in range(start, end+1)], 1)[0]
                data = self._get_data(batch_size=50, task_params=(selected, ))
            elif self.meta_params["data_type"] == "distribution":
                selected = self.task_set[i]
                data = self._get_data(batch_size=50, task_params=selected)
            else:
                raise NotImplementedError
            model_score = self._fast_val(self.meta_model, data=data, mode="eval", return_all=True)
            model_score = model_score.tolist()

            if self.meta_params["solver"] == "lkh3":
                # get results from LKH3 (~14s)
                opts = argparse.ArgumentParser()
                opts.cpus, opts.n, opts.progress_bar_mininterval = None, None, 0.1
                dataset = [(instance.cpu().numpy(),) for instance in data]
                executable = get_lkh_executable()
                global run_func
                def run_func(args):
                    return solve_lkh_log(executable, *args, runs=1, disable_cache=True)  # otherwise it directly loads data from dir
                results, _ = run_all_in_pool(run_func, "./LKH3_result", dataset, opts, use_multiprocessing=False)
                gap_list = [(model_score[j]-results[j][0])/results[j][0]*100 for j in range(len(results))]
                gap[i] = sum(gap_list)/len(gap_list)
            elif self.meta_params["solver"] == "best_model":
                best_model_score = self._fast_val(self.best_meta_model, data=data, mode="eval", return_all=True)
                best_model_score = best_model_score.tolist()
                gap_list = [(model_score[j] - best_model_score[j]) / best_model_score[j] * 100 for j in range(len(best_model_score))]
                gap[i] = sum(gap_list) / len(gap_list)
            elif self.meta_params["solver"] == "bootstrap":
                bootstrap_reward = self._bootstrap(copy.deepcopy(self.meta_model), data, mode="eval")
                bootstrap_score = (-bootstrap_reward).view(-1).float().tolist()
                gap_list = [(model_score[j] - bootstrap_score[j]) / bootstrap_score[j] * 100 for j in range(len(bootstrap_score))]
                gap[i] = sum(gap_list) / len(gap_list)
            else:
                raise NotImplementedError
        print(">> Finish updating task weights within {}s".format(round(time.time()-start_t, 2)))

        # temp = max(1.0 * (1 - epoch / self.meta_params["epochs"]), 0.05)
        # temp = max(1.0 - 1/2 * (1 - math.cos(math.pi * min(epoch / self.meta_params['epochs'], 1))), 0.2)
        temp = 0.25
        gap_temp = torch.Tensor([i/temp for i in gap.tolist()])
        print(gap, temp)
        print(">> Old task weights: {}".format(self.task_w))
        self.task_w = torch.softmax(gap_temp, dim=0)
        print(">> New task weights: {}".format(self.task_w))
