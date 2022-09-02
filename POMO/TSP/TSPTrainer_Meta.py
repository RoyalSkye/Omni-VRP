import copy
import math
import random
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from TSProblemDef import get_random_problems

from utils.utils import *


class TSPTrainer:
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
        self.alpha = self.meta_params['alpha']
        if self.meta_params['data_type'] == "distribution":  # focus on the TSP100 with different distributions
            self.task_set = [(m, l) for l in [1, 10, 20, 30, 50] for m in range(1, 1+self.meta_params['num_task']//5)] + [(0, 0)]
        elif self.meta_params['data_type'] == "size":  # focus on uniform distribution with different sizes
            self.task_set = [(n, ) for n in range(5, 5 + 5 * self.meta_params['num_task'], 5)]
        elif self.meta_params['data_type'] == "size_distribution":
            task_set = [(m, l) for l in [1, 10, 20, 30, 50] for m in range(1, 11)] + [(0, 0)]
            self.task_set = [(n, m, l) for n in [25, 50, 75, 100, 125, 150] for (m, l) in task_set]
        else:
            raise NotImplementedError
        print(">> Training task set: {}".format(self.task_set))
        assert self.trainer_params['meta_params']['epochs'] == math.ceil((1000 * 100000) / (
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
        """
        1. Sample B training tasks from task distribution P(T)
        2. for each of task T_i, do reptile -> \theta_i
        3. update meta-model \theta_0
        """
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.meta_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            if self.meta_params['meta_method'] == 'reptile':
                train_score, train_loss = self._train_one_epoch(epoch)
            else:
                raise NotImplementedError
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
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

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        batch_size = self.meta_params['meta_batch_size']
        self._alpha_scheduler(epoch)
        slow_weights = copy.deepcopy(self.meta_model.state_dict())
        fast_weights = []

        for i in range(self.meta_params['B']):
            task_params = random.sample(self.task_set, 1)[0]  # uniform sample a task
            task_model = copy.deepcopy(self.meta_model)
            optimizer = Optimizer(task_model.parameters(), **self.optimizer_params['optimizer'])
            env_params = {
                'problem_size': task_params,
                'pomo_size': task_params,
            } if self.meta_params['data_type'] != 'distribution' else self.env_params
            env = Env(**env_params)

            for batch_id in range(self.meta_params['k']):
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
                avg_score, avg_loss = self._train_one_batch(task_model, data, optimizer, env)
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)

            fast_weights.append(task_model.state_dict())

        state_dict = {params_key: (slow_weights[params_key] + self.alpha * torch.mean(torch.stack([fast_weight[params_key] - slow_weights[params_key] for fast_weight in fast_weights], dim=0), dim=0))
                      for params_key in slow_weights}
        self.meta_model.load_state_dict(state_dict)

        # Log Once, for each epoch
        self.logger.info('Meta Iteration {:3d}: alpha: {:6f}, Score: {:.4f},  Loss: {:.4f}'.format(epoch, self.alpha, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, task_model, data, optimizer, env):

        # Prep
        task_model.train()
        batch_size = data.size(0)
        env.load_problems(batch_size, problems=data)
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

        # Step & Return
        task_model.zero_grad()
        loss_mean.backward()
        optimizer.step()

        return score_mean.item(), loss_mean.item()

    def _fast_val(self, task_model, data, env):
        """
        TODO: a simple implementation of fast evaluation at the end of each meta training iteration.
        """
        return 0, 0

    def _alpha_scheduler(self, iter):
        self.alpha *= self.meta_params['alpha_decay']
