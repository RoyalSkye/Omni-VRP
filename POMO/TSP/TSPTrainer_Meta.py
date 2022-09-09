import copy
import math
import random
import torch
from logging import getLogger
from collections import OrderedDict

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from TSProblemDef import get_random_problems, generate_task_set

from utils.utils import *
from utils.functions import load_dataset


class TSPTrainer:
    """
    Implementation of POMO with MAML / FOMAML / Reptile.
    For MAML & FOMAML, ref to "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks";
    For Reptile, ref to "On First-Order Meta-Learning Algorithms".
    """
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.meta_params = trainer_params['meta_params']

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
        self.alpha = self.meta_params['alpha']  # for reptile
        self.task_set = generate_task_set(self.meta_params)
        assert self.trainer_params['meta_params']['epochs'] == math.ceil((self.trainer_params['epochs'] * self.trainer_params['train_episodes']) / (
                    self.trainer_params['meta_params']['B'] * self.trainer_params['meta_params']['k'] *
                    self.trainer_params['meta_params']['meta_batch_size'])), ">> meta-learning iteration does not match with POMO!"

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)
        val_res = []
        for epoch in range(self.start_epoch, self.meta_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.meta_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}({:.2f}%): Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.meta_params['epochs'], epoch/self.meta_params['epochs']*100, elapsed_time_str, remain_time_str))

            all_done = (epoch == self.meta_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                # val
                val_episodes = 256
                _, no_aug_score = self._fast_val(copy.deepcopy(self.meta_model), val_episodes=val_episodes)
                val_res.append(no_aug_score)
                print(">> validation results: {} over {} instances".format(no_aug_score, val_episodes))
                # save checkpoint
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.meta_model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                print(val_res)

    def _train_one_epoch(self, epoch):
        """
        1. Sample B training tasks from task distribution P(T)
        2. inner-loop: for a batch of tasks T_i, do reptile -> \theta_i
        3. outer-loop: update meta-model -> \theta_0
        """
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        batch_size = self.meta_params['meta_batch_size']

        self._alpha_scheduler(epoch)
        slow_weights = copy.deepcopy(self.meta_model.state_dict())
        fast_weights, val_loss, fomaml_grad = [], 0, []

        # sample a batch of tasks
        for i in range(self.meta_params['B']):
            task_params = random.sample(self.task_set, 1)[0]
            task_model = copy.deepcopy(self.meta_model)

            for step in range(self.meta_params['k'] + 1):
                # generate task-specific data
                if self.meta_params['data_type'] == 'distribution':
                    assert len(task_params) == 2
                    data = get_random_problems(batch_size, self.env_params['problem_size'], num_modes=task_params[0], cdist=task_params[-1], distribution='gaussian_mixture')
                elif self.meta_params['data_type'] == 'size':
                    assert len(task_params) == 1
                    data = get_random_problems(batch_size, task_params[0], num_modes=0, cdist=0, distribution='uniform')
                elif self.meta_params['data_type'] == "size_distribution":
                    assert len(task_params) == 3
                    data = get_random_problems(batch_size, problem_size=task_params[0], num_modes=task_params[1], cdist=task_params[-1], distribution='gaussian_mixture')
                else:
                    raise NotImplementedError

                if step == self.meta_params['k']: continue
                env_params = {'problem_size': data.size(1), 'pomo_size': data.size(1)}
                avg_score, avg_loss = self._train_one_batch(step, task_model, data, Env(**env_params))
                score_AM.update(avg_score.item(), batch_size)
                loss_AM.update(avg_loss.item(), batch_size)

            if self.meta_params['meta_method'] == 'maml':
                # cal loss on query(val) set - data
                val_loss += self._fast_val(task_model, data=data)[0]
            elif self.meta_params['meta_method'] == 'fomaml':
                val_loss = self._fast_val(task_model, data=data)[0]
                task_model.train()
                grad = torch.autograd.grad(val_loss, task_model.parameters())
                fomaml_grad.append(grad)
            elif self.meta_params['meta_method'] == 'reptile':
                fast_weights.append(task_model.state_dict())

        # update meta-model
        if self.meta_params['meta_method'] == 'maml':
            val_loss = val_loss / self.meta_params['B']
            gradients = torch.autograd.grad(val_loss, self.maml)
            updated_weights = OrderedDict(
                (name, param - self.optimizer_params['optimizer']['lr'] * grad)
                for ((name, param), grad) in zip(self.meta_model.state_dict().items(), gradients)
            )
            self.meta_model.load_state_dict(updated_weights)
        elif self.meta_params['meta_method'] == 'fomaml':
            updated_weights = self.meta_model.state_dict()
            for gradients in fomaml_grad:
                updated_weights = OrderedDict(
                    (name, param - self.optimizer_params['optimizer']['lr'] / self.meta_params['B'] * grad)
                    for ((name, param), grad) in zip(updated_weights.items(), gradients)
                )
            self.meta_model.load_state_dict(updated_weights)
        elif self.meta_params['meta_method'] == 'reptile':
            state_dict = {params_key: (slow_weights[params_key] + self.alpha * torch.mean(torch.stack([fast_weight[params_key] - slow_weights[params_key] for fast_weight in fast_weights], dim=0), dim=0)) for params_key in slow_weights}
            self.meta_model.load_state_dict(state_dict)

        # Log Once, for each epoch
        self.logger.info('Meta Iteration {:3d}: alpha: {:6f}, Score: {:.4f},  Loss: {:.4f}'.format(epoch, self.alpha, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, i, task_model, data, env):

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

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # update model
        create_graph = True if self.meta_params['meta_method'] == 'maml' else False
        if i == 0:
            self.maml = list(task_model.parameters())
            gradients = torch.autograd.grad(loss_mean, self.maml, create_graph=create_graph)
        else:
            gradients = torch.autograd.grad(loss_mean, task_model.parameters(), create_graph=create_graph)
        fast_weights = OrderedDict(
            (name, param - self.optimizer_params['optimizer']['lr'] * grad)
            for ((name, param), grad) in zip(task_model.state_dict().items(), gradients)
        )
        task_model.load_state_dict(fast_weights)

        return score_mean, loss_mean

    def _fast_val(self, model, data=None, val_episodes=256):
        aug_factor = 1
        if data is None:
            val_path = "../../data/TSP/tsp50_tsplib.pkl"
            data = torch.Tensor(load_dataset(val_path)[: val_episodes])
        env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})

        model.eval()
        batch_size = data.size(0)
        with torch.enable_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))

            state, reward, done = env.pre_step()
            while not done:
                selected, prob = model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)  # for the first/last node, p=1 -> log_p=0
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        return loss_mean, no_aug_score.detach().item()

    def _alpha_scheduler(self, iter):
        self.alpha = max(self.alpha * self.meta_params['alpha_decay'], 0.0001)
