
import torch

import os
from logging import getLogger
from torch.optim import Adam as Optimizer

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from TSP_gurobi import solve_all_gurobi
from utils.utils import *
from utils.functions import load_dataset, save_dataset


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.fine_tune_params = tester_params['fine_tune_params']

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # ENV and MODEL
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)  # we assume instances in the test/fine-tune dataset have the same problem size.
        self.optimizer = Optimizer(self.model.parameters(), **self.tester_params['fine_tune_params']['optimizer'])

        # load dataset
        self.test_data = load_dataset(tester_params['test_set_path'])[: self.tester_params['test_episodes']]
        opt_sol = load_dataset(tester_params['test_set_opt_sol_path'])[: self.tester_params['test_episodes']]  # [(obj, route), ...]
        self.opt_sol = [i[0] for i in opt_sol]
        if self.fine_tune_params['enable']:
            start = tester_params['test_episodes'] if self.tester_params['test_set_path'] == self.fine_tune_params['fine_tune_set_path'] else 0
            self.fine_tune_data = load_dataset(self.fine_tune_params['fine_tune_set_path'])[start: start+self.fine_tune_params['fine_tune_episodes']]
        else:
            self.fine_tune_data = None

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # TODO: which performance is good? load or not load?
        self.logger.info(">> Model loaded from {}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        if self.tester_params['test_robustness']:
            episode = 0
            test_data = torch.Tensor(self.test_data)
            opt_sol = [0] * test_data.size(0)
            while episode < test_data.size(0):
                remaining = test_data.size(0) - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)
                data = torch.Tensor(test_data[episode: episode + batch_size])
                test_data[episode: episode + batch_size], opt_sol[episode: episode + batch_size] = self._generate_x_adv(data, eps=50.0)
                episode += batch_size
            self.test_data = test_data.cpu().numpy()
            self.opt_sol = [i[0] for i in opt_sol]
            # save the adv dataset
            filename = os.path.split(self.tester_params['test_set_path'])[-1]
            save_dataset(self.test_data, './adv_{}'.format(filename))
            save_dataset(opt_sol, './sol_adv_{}'.format(filename))
        if self.fine_tune_params['enable']:
            # fine-tune model on fine-tune dataset (few-shot)
            self._fine_tune_and_test()
        else:
            # test the model on test dataset
            self._test()

    def _test(self):

        self.time_estimator.reset()
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        assert len(self.test_data) == test_num_episode, "the number of test instances does not match!"
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            score, aug_score, all_score, all_aug_score = self._test_one_batch(torch.Tensor(self.test_data[episode: episode + batch_size]))
            opt_sol = self.opt_sol[episode: episode + batch_size]
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            episode += batch_size
            gap = [max(all_score[i].item() - opt_sol[i], 0) / opt_sol[i] * 100 for i in range(batch_size)]
            aug_gap = [max(all_aug_score[i].item() - opt_sol[i], 0) / opt_sol[i] * 100 for i in range(batch_size)]
            gap_AM.update(sum(gap)/batch_size, batch_size)
            aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))

        return score_AM.avg, aug_score_AM.avg, gap_AM.avg, aug_gap_AM.avg

    def _test_one_batch(self, test_data):
        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        batch_size = test_data.size(0)
        with torch.no_grad():
            self.env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score

    def _fine_tune_and_test(self):
        """
        evaluate few-shot generalization: fine-tune k steps on a small fine-tune dataset, test on test dataset after every step
        """
        fine_tune_episode = self.fine_tune_params['fine_tune_episodes']
        assert len(self.fine_tune_data) == fine_tune_episode, "the number of fine-tune instances does not match!"
        score_list, aug_score_list, gap_list, aug_gap_list = [], [], [], []
        score, aug_score, gap, aug_gap = self._test()
        score_list.append(score); aug_score_list.append(aug_score)
        gap_list.append(gap); aug_gap_list.append(aug_gap)

        for k in range(self.fine_tune_params['k']):
            self.logger.info("Start fine-tune step {}".format(k+1))
            episode = 0
            while episode < fine_tune_episode:
                remaining = fine_tune_episode - episode
                batch_size = min(self.fine_tune_params['fine_tune_batch_size'], remaining)
                self._fine_tune_one_batch(torch.Tensor(self.fine_tune_data[episode:episode+batch_size]))
                episode += batch_size
            score, aug_score, gap, aug_gap = self._test()
            score_list.append(score); aug_score_list.append(aug_score)
            gap_list.append(gap); aug_gap_list.append(aug_gap)

        print(self.tester_params['test_set_path'])
        print("Final score_list: {}".format(score_list))
        print("Final aug_score_list {}".format(aug_score_list))
        print("Final gap_list: {}".format(gap_list))
        print("Final aug_gap_list: {}".format(aug_gap_list))

    def _fine_tune_one_batch(self, fine_tune_data):
        # Augmentation
        if self.fine_tune_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.model.train()
        batch_size = fine_tune_data.size(0)
        self.env.load_problems(batch_size, problems=fine_tune_data, aug_factor=aug_factor)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(aug_factor * batch_size, self.env.pomo_size, 0))
        # shape: (augmentation * batch, pomo, 0~problem)

        # POMO Rollout, please note that the reward is negative (i.e., -length of route).
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (augmentation * batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).view(batch_size, -1)
        # shape: (batch, augmentation * pomo)
        advantage = aug_reward - aug_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, augmentation * pomo)
        log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).view(batch_size, -1)
        # size = (batch, augmentation * pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, augmentation * pomo)
        loss_mean = loss.mean()

        # Score
        max_pomo_reward, _ = aug_reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

    def _generate_x_adv(self, data, eps=10.0):
        """
        Generate adversarial data based on the current model, also need to generate optimal sol for x_adv.
        """
        from torch.autograd import Variable
        def minmax(xy_):
            # min_max normalization: [b,n,2]
            xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
            return xy_

        # generate x_adv
        self.model.eval()
        aug_factor, batch_size = 1, data.size(0)
        with torch.enable_grad():
            data.requires_grad_()
            self.env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            # print(self.env.problems.requires_grad)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            prob_list = torch.zeros(size=(aug_factor * batch_size, self.env.pomo_size, 0))
            state, reward, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).view(batch_size, -1)
            baseline_reward = aug_reward.float().mean(dim=1, keepdims=True)
            advantage = aug_reward - baseline_reward
            log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).view(batch_size, -1)

            # delta = torch.autograd.grad(eps * ((advantage / baseline_reward) * log_prob).mean(), data)[0]
            delta = torch.autograd.grad(eps * ((-advantage) * log_prob).mean(), data)[0]
            data = data.detach() + delta
            data = minmax(data)
            data = Variable(data, requires_grad=False)

        # generate opt sol
        opt_sol = solve_all_gurobi(data)

        return data, opt_sol
