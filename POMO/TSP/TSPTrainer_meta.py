import os
import copy
import math
import time
import random
import torch
from logging import getLogger
from collections import OrderedDict
from torch.optim import Adam as Optimizer
# from torch.optim import SGD as Optimizer

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from ProblemDef import get_random_problems, generate_task_set
from utils.utils import *
from utils.functions import *
from TSP_baseline import *


class TSPTrainer:
    """
    Implementation of POMO with MAML / FOMAML / Reptile / Bootstrap Meta-learning on TSP.
    For MAML & FOMAML, ref to "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks";
    For Reptile, ref to "On First-Order Meta-Learning Algorithms" and "On the generalization of neural combinatorial optimization heuristics".
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
        # (`no` norm) and (`batch` with fomaml) will destabilize the meta-training, while (batch with maml) is ok;
        # On the zero-shot setting, `instance` norm and `batch_no_track` are better than `batch` norm;
        # On the few-shot setting, `batch` norm seems to better than `instance` norm, with a faster adaptation to OOD data.
        self.model_params["norm"] = 'batch_no_track'
        self.meta_model = Model(**self.model_params)
        self.meta_optimizer = Optimizer(self.meta_model.parameters(), **self.optimizer_params['optimizer'])
        self.alpha = self.meta_params['alpha']  # for reptile
        self.early_stop = True if self.meta_params['meta_method'] == 'maml_fomaml' else False
        self.task_set = generate_task_set(self.meta_params)
        self.val_data, self.val_opt = {}, {}  # for lkh3_offline
        if self.meta_params["data_type"] == "size_distribution":
            # hardcoded - task_set: range(self.min_n, self.max_n, self.task_interval) * self.num_dist
            self.min_n, self.max_n, self.task_interval, self.num_dist = 50, 200, 5, 11
            self.task_w = torch.full(((self.max_n - self.min_n) // self.task_interval + 1, self.num_dist), 1 / self.num_dist)

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        pretrain_load = trainer_params['pretrain_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Checkpoint loaded successfully from {}'.format(checkpoint_fullname))
            
        elif pretrain_load['enable']:  # meta-training on a pretrained model
            self.logger.info(">> Loading pretrained model: be careful with the type of the normalization layer!")
            checkpoint_fullname = '{path}'.format(**pretrain_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # otherwise, unstable meta-training (nan problem)
            self.logger.info('Pretrained model loaded successfully from {}'.format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        start_time, best_mean = time.time(), 1000
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.meta_params['epochs']+1):
            self.logger.info('=================================================================')

            # lr decay (by 10) to speed up convergence at 90th iteration
            if epoch in [int(self.meta_params['epochs'] * 0.9)]:
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
            if self.meta_params["data_type"] == "size_distribution":
                dir = "../../data/TSP/Size_Distribution/"
                paths = ["tsp200_uniform.pkl", "tsp300_rotation.pkl"]
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
                epoch, self.meta_params['epochs'], epoch/self.meta_params['epochs']*100, elapsed_time_str, remain_time_str, no_aug_score_list))

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
        Meta-Learning framework:
            1. Sample B training tasks from task distribution P(T)
            2. Inner-loop: for a batch of tasks T_i, POMO training -> \theta_i
            3. Outer-loop: update meta-model -> \theta_0

        Adaptive task scheduler:
            for size: gradually increase the problem size (Curriculum learning);
            for distribution: we compute the relative gaps (w.r.t. LKH3) or estimate the potential improvements of each distribution (i.e., bootstrap) every X iters;
            for size_distribution: combine together.
        """
        self.meta_optimizer.zero_grad()
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        meta_batch_size = self.meta_params['meta_batch_size']
        if self.early_stop:
            if epoch > self.meta_params['early_stop_epoch']:
                self.meta_params['meta_method'] = 'fomaml'
            else:
                self.meta_params['meta_method'] = 'maml'

        # Adaptive task scheduler:
        if self.meta_params['curriculum']:
            if self.meta_params["data_type"] == "size_distribution":
                start = self.min_n + int(min(epoch / self.meta_params['sch_epoch'], 1) * (self.max_n - self.min_n))  # linear
                # start = self.min_n + int(1 / 2 * (1 - math.cos(math.pi * min(epoch / self.meta_params['sch_epoch'], 1))) * (self.max_n - self.min_n))  # cosine
                n = start // self.task_interval * self.task_interval
                idx = (n - self.min_n) // self.task_interval
                tasks, weights = self.task_set[idx*11: (idx+1)*11], self.task_w[idx]
                if epoch % self.meta_params['update_weight'] == 0:
                    self.task_w[idx] = self._update_task_weight(tasks, weights, epoch)

        self._alpha_scheduler(epoch)  # for reptile
        fast_weights, val_loss, meta_grad_dict = [], 0, {(i, j): 0 for i, group in enumerate(self.meta_optimizer.param_groups) for j, _ in enumerate(group['params'])}

        # sample a batch of tasks
        w, selected_tasks = [1.0] * self.meta_params['B'], []
        for b in range(self.meta_params['B']):
            if self.meta_params["data_type"] == "size_distribution":
                if self.meta_params['curriculum']:
                    selected = torch.multinomial(self.task_w[idx], 1).item()
                    task_params = tasks[selected]
                    w[b] = self.task_w[idx][selected].item()
                else:
                    task_params = random.sample(self.task_set, 1)[0]
                batch_size = meta_batch_size if task_params[0] <= 150 else meta_batch_size // 2
            selected_tasks.append(task_params)
        w = torch.softmax(torch.Tensor(w), dim=0)

        for b in range(self.meta_params['B']):
            task_params, task_w = selected_tasks[b], w[b].item()
            # preparation
            if self.meta_params['meta_method'] in ['fomaml', 'reptile']:
                task_model = copy.deepcopy(self.meta_model)
                optimizer = Optimizer(task_model.parameters(), **self.optimizer_params['optimizer'])
                # optimizer.load_state_dict(self.meta_optimizer.state_dict())  # may cause unstable meta-training for fomaml
            elif self.meta_params['meta_method'] == 'maml':
                if self.model_params['meta_update_encoder']:
                    fast_weight = OrderedDict(self.meta_model.named_parameters())
                else:
                    fast_weight = OrderedDict(self.meta_model.decoder.named_parameters())
                    for k in list(fast_weight.keys()):
                        fast_weight["decoder."+k] = fast_weight.pop(k)

            # inner-loop optimization
            for step in range(self.meta_params['k']):
                data = self._get_data(batch_size, task_params)
                env_params = {'problem_size': data.size(1), 'pomo_size': data.size(1)}
                self.meta_model.train()
                if self.meta_params['meta_method'] in ['reptile', 'fomaml']:
                    avg_score, avg_loss = self._train_one_batch(task_model, data, Env(**env_params), optimizer)
                elif self.meta_params['meta_method'] == 'maml':
                    avg_score, avg_loss, fast_weight = self._train_one_batch_maml(fast_weight, data, Env(**env_params), create_graph=True)
                score_AM.update(avg_score.item(), batch_size)
                loss_AM.update(avg_loss.item(), batch_size)

            # bootstrap
            bootstrap_model = None
            if self.meta_params['L'] > 0:
                assert self.meta_params['meta_method'] in ['maml', 'fomaml']
                bootstrap_model = Model(**self.model_params)
                if self.meta_params['meta_method'] == 'maml':
                    bootstrap_model = OrderedDict({k: v.clone().detach().requires_grad_(True) for k, v in fast_weight.items()})
                else:
                    bootstrap_model.load_state_dict(copy.deepcopy(task_model.state_dict()))
                    bootstrap_optimizer = Optimizer(bootstrap_model.parameters(), **self.optimizer_params['optimizer'])
                    bootstrap_optimizer.load_state_dict(optimizer.state_dict())
            for step in range(self.meta_params['L']):
                data = self._get_data(batch_size, task_params)
                if self.meta_params['meta_method'] == 'maml':
                    avg_score, avg_loss, bootstrap_model = self._train_one_batch_maml(bootstrap_model, data, Env(**env_params), create_graph=False)
                else:
                    avg_score, avg_loss = self._train_one_batch(bootstrap_model, data, Env(**env_params), bootstrap_optimizer)

            val_data = self._get_val_data(batch_size, task_params)
            self.meta_model.train()
            if self.meta_params['meta_method'] == 'maml':
                # Old version
                # val_loss += self._fast_val(fast_weight, data=val_data, mode="maml") / self.meta_params['B']
                # New version - Save GPU memory
                val_loss, kl_loss = self._fast_val(fast_weight, data=val_data, mode="maml", bootstrap_model=bootstrap_model)
                print(val_loss, kl_loss)
                loss = (self.meta_params['beta'] * val_loss + (1-self.meta_params['beta']) * kl_loss) * task_w
                self.meta_optimizer.zero_grad()
                loss.backward()
                for i, group in enumerate(self.meta_optimizer.param_groups):
                    for j, p in enumerate(group['params']):
                        meta_grad_dict[(i, j)] += p.grad
            elif self.meta_params['meta_method'] == 'fomaml':
                val_loss, kl_loss = self._fast_val(task_model, data=val_data, mode="fomaml", bootstrap_model=bootstrap_model)
                print(val_loss, kl_loss)
                loss = (self.meta_params['beta'] * val_loss + (1-self.meta_params['beta']) * kl_loss) * task_w
                optimizer.zero_grad()
                loss.backward()
                for i, group in enumerate(optimizer.param_groups):
                    for j, p in enumerate(group['params']):
                        meta_grad_dict[(i, j)] += p.grad
            elif self.meta_params['meta_method'] == 'reptile':
                fast_weights.append(task_model.state_dict())

        # outer-loop optimization (update meta-model)
        if self.meta_params['meta_method'] == 'maml':
            # Old version
            # self.meta_optimizer.zero_grad()
            # val_loss.backward()
            # self.meta_optimizer.step()
            # New version - Save GPU memory
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
            state_dict = {params_key: (self.meta_model.state_dict()[params_key] + self.alpha * torch.mean(torch.stack([fast_weight[params_key] - self.meta_model.state_dict()[params_key] for fast_weight in fast_weights], dim=0).float(), dim=0)) for params_key in self.meta_model.state_dict()}
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

    def _train_one_batch_maml(self, fast_weight, data, env, optimizer=None, create_graph=True):

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

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # 1. update model - in SGD way
        # gradients = torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=create_graph)  # allow_unused=True
        # fast_weight = OrderedDict(
        #     (name, param - self.optimizer_params['optimizer']['lr'] * grad)
        #     for ((name, param), grad) in zip(fast_weight.items(), gradients)
        # )
        # 2. update model - in Adam way
        gradients = torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=create_graph)  # allow_unused=True
        w_t, (beta1, beta2), eps = [], self.meta_optimizer.param_groups[0]['betas'], self.meta_optimizer.param_groups[0]['eps']
        lr, weight_decay = self.optimizer_params['optimizer']['lr'], self.optimizer_params['optimizer']['weight_decay']
        for i, ((name, param), grad) in enumerate(zip(fast_weight.items(), gradients)):
            if self.meta_optimizer.state_dict()['state'] != {}:
                # (with batch/instnace norm layer): i \in [0, 85], where encoder \in [0, 79] + decoder \in [80, 85]
                # (with rezero norm layer): i \in [0, 73], where encoder \in [0, 67] + decoder \in [68, 73]
                # (without norm layer): i \in [0, 61], where encoder \in [0, 55] + decoder \in [56, 61]
                i = i if self.model_params['meta_update_encoder'] else i + 80
                state = self.meta_optimizer.state_dict()['state'][i]
                step, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']
                step += 1
                step = step.item() if isinstance(step, torch.Tensor) else step
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
        """
        # 3. update model using optimizer - this method can not work properly.
        optimizer.zero_grad()
        # torch.autograd.grad(loss_mean, fast_weight.values(), create_graph=create_graph)
        # print(list(self.meta_model.parameters())[-1])
        loss_mean.backward(retain_graph=True, create_graph=True)
        optimizer.step()  # will update meta_model as well...
        """

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        print(score_mean)

        return score_mean, loss_mean, fast_weight

    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=32, mode="eval", return_all=False, bootstrap_model=None):
        aug_factor = 1
        data = torch.Tensor(load_dataset(path)[offset: offset+val_episodes]) if data is None else data
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
            fast_weight, kl_loss = model, 0
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            if mode == "maml":
                self.meta_model.pre_forward(reset_state, weights=fast_weight)
                if bootstrap_model is not None:
                    with torch.no_grad():
                        self.meta_model.pre_forward(reset_state, weights=bootstrap_model)
            else:
                model.pre_forward(reset_state)
                if bootstrap_model is not None:
                    with torch.no_grad():
                        bootstrap_model.pre_forward(reset_state)

            prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
            state, reward, done = env.pre_step()
            while not done:
                if mode == "maml":
                    selected, prob, probs = self.meta_model(state, weights=fast_weight, return_probs=True)
                    if bootstrap_model is not None:
                        probs1 = torch.where(probs > 0, probs, torch.tensor(0.00001))
                        with torch.no_grad():
                            _, _, bs_probs = self.meta_model(state, weights=bootstrap_model, selected=selected, return_probs=True)
                            bs_probs = torch.where(bs_probs > 0, bs_probs, torch.tensor(0.00001))
                else:
                    selected, prob, probs = model(state, return_probs=True)
                    if bootstrap_model is not None:
                        probs1 = torch.where(probs > 0, probs, torch.tensor(0.00001))
                        with torch.no_grad():
                            _, _, bs_probs = bootstrap_model(state, selected=selected, return_probs=True)
                            bs_probs = torch.where(bs_probs > 0, bs_probs, torch.tensor(0.00001))

                # shape: (batch, pomo)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
                kl_loss += (bs_probs * (bs_probs.log() - probs1.log())).reshape(batch_size * data.size(1), -1).sum(dim=-1).mean() if bootstrap_model is not None else 0

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
            return loss_mean, kl_loss

    def _get_data(self, batch_size, task_params):
        if self.meta_params['data_type'] == 'size':
            assert len(task_params) == 1
            data = get_random_problems(batch_size, task_params[0], num_modes=0, cdist=0, distribution='uniform', problem="tsp")
        elif self.meta_params['data_type'] == 'distribution':
            assert len(task_params) == 2
            data = get_random_problems(batch_size, self.env_params['problem_size'], num_modes=task_params[0], cdist=task_params[1], distribution='gaussian_mixture', problem="tsp")
        elif self.meta_params['data_type'] == "size_distribution":
            assert len(task_params) == 3
            data = get_random_problems(batch_size, task_params[0], num_modes=task_params[1], cdist=task_params[2], distribution='gaussian_mixture', problem="tsp")
        else:
            raise NotImplementedError

        return data

    def _get_val_data(self, batch_size, task_params):
        if self.meta_params["data_type"] == "size":
            val_data = self._get_data(batch_size, task_params)
        elif self.meta_params["data_type"] == "distribution":
            val_data = self._get_data(batch_size, task_params)
        elif self.meta_params["data_type"] == "size_distribution":
            val_data = self._get_data(batch_size, task_params)
        else:
            raise NotImplementedError

        return val_data

    def _alpha_scheduler(self, epoch):
        """
        Update param for Reptile.
        """
        self.alpha = max(self.alpha * self.meta_params['alpha_decay'], 0.0001)

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
            data = self._get_data(batch_size=batch_size, task_params=selected)

            # only use lkh3 at the first iteration of updating task weights
            if self.meta_params["solver"] == "lkh3_offline":
                if selected not in self.val_data.keys():
                    self.val_data[selected] = data
                    opts = argparse.ArgumentParser()
                    opts.cpus, opts.n, opts.progress_bar_mininterval = None, None, 0.1
                    dataset = [(instance.cpu().numpy(),) for instance in data]
                    executable = get_lkh_executable()
                    def run_func(args):
                        return solve_lkh_log(executable, *args, runs=1, disable_cache=True, MAX_TRIALS=100)  # otherwise it directly loads data from dir
                    results, _ = run_all_in_pool(run_func, "./LKH3_result", dataset, opts, use_multiprocessing=False)
                    self.val_opt[selected] = [j[0] for j in results]
                data = self.val_data[selected][idx]

            model_score = self._fast_val(self.meta_model, data=data, mode="eval", return_all=True)
            model_score = model_score.tolist()

            if self.meta_params["solver"] == "lkh3_online":
                # get results from LKH3
                opts = argparse.ArgumentParser()
                opts.cpus, opts.n, opts.progress_bar_mininterval = None, None, 0.1
                dataset = [(instance.cpu().numpy(),) for instance in data]
                executable = get_lkh_executable()
                def run_func(args):
                    return solve_lkh_log(executable, *args, runs=1, disable_cache=True, MAX_TRIALS=100)  # otherwise it directly loads data from dir
                results, _ = run_all_in_pool(run_func, "./LKH3_result", dataset, opts, use_multiprocessing=False)
                gap_list = [(model_score[j]-results[j][0])/results[j][0]*100 for j in range(len(results))]
                gap[i] = sum(gap_list)/len(gap_list)
            elif self.meta_params["solver"] == "lkh3_offline":
                lkh_score = [self.val_opt[selected][j] for j in idx.tolist()]
                gap_list = [(model_score[j] - lkh_score[j]) / lkh_score[j] * 100 for j in range(len(lkh_score))]
                gap[i] = sum(gap_list) / len(gap_list)
            elif self.meta_params["solver"] == "best_model":  # not recommend: how to define the best model? (biased to the val dataset)
                best_model_score = self._fast_val(self.best_meta_model, data=data, mode="eval", return_all=True)
                best_model_score = best_model_score.tolist()
                gap_list = [(model_score[j] - best_model_score[j]) / best_model_score[j] * 100 for j in range(len(best_model_score))]
                gap[i] = sum(gap_list) / len(gap_list)
            else:
                raise NotImplementedError
        print(">> Finish updating task weights within {}s".format(round(time.time()-start_t, 2)))

        temp = 1.0
        gap_temp = torch.Tensor([i/temp for i in gap.tolist()])
        print(gap, temp)
        print(">> Old task weights: {}".format(weights))
        weights = torch.softmax(gap_temp, dim=0)
        print(">> New task weights: {}".format(weights))

        return weights

    def _get_kl_loss(self, bootstrap_model, val_data, slow_tour, slow_probs):
        """
        Ref to "Bootstrap Meta-Learning", ICLR 2022;
        This function is deprecated since
            a. storing probs_list for large-scale COPs on GPU is extremely (memory) expensive (e.g., > 20GB on TSP200);
            b. probs_list.cpu() at every step is also extremely (time) expensive.
        Instead, we compute KL loss on the fly now, see self._fast_val()
        """
        if isinstance(bootstrap_model, torch.nn.Module):
            bootstrap_model.eval()
        env = Env(**{'problem_size': val_data.size(1), 'pomo_size': val_data.size(1)})
        batch_size = val_data.size(0)
        env.load_problems(batch_size, problems=val_data, aug_factor=1)
        reset_state, _, _ = env.reset()

        with torch.no_grad():
            if self.meta_params['meta_method'] == 'maml':
                self.meta_model.pre_forward(reset_state, weights=bootstrap_model)
            else:
                bootstrap_model.pre_forward(reset_state)
            probs_list = torch.zeros(size=(batch_size, env.pomo_size, env.problem_size, 0))
            state, reward, done = env.pre_step()
            selected_idx = 0
            while not done:
                if self.meta_params['meta_method'] == 'maml':
                    selected, prob, probs = self.meta_model(state, weights=bootstrap_model, selected=slow_tour[:, :, selected_idx].reshape(batch_size, -1).long(), return_probs=True)
                else:
                    selected, prob, probs = bootstrap_model(state, selected=slow_tour[:, :, selected_idx].reshape(batch_size, -1).long(), return_probs=True)
                # shape: (batch, pomo)
                selected_idx += 1
                state, reward, done = env.step(selected)
                probs_list = torch.cat((probs_list, probs[:, :, :, None]), dim=3)
            probs_list = torch.where(probs_list > 0, probs_list, torch.tensor(0.00001))

        slow_probs = torch.where(slow_probs > 0, slow_probs, torch.tensor(0.00001))  # avoid log0
        # kl_loss = (probs_list * (probs_list.log() - slow_probs.log())).sum(dim=2).mean()
        kl_loss = (probs_list * (probs_list.log() - slow_probs.log())).reshape(batch_size * val_data.size(1), -1).sum(dim=-1).mean()
        # kl_loss = torch.nn.KLDivLoss(reduction="batchmean")(slow_probs.log().reshape(batch_size * val_data.size(1), -1), probs_list.reshape(batch_size * val_data.size(1), -1))

        return kl_loss
