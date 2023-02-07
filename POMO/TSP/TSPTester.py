import os, time
import pickle
from logging import getLogger
import numpy as np
import torch
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
                 tester_params,
                 fine_tune_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.fine_tune_params = fine_tune_params

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
        self.path_list = None
        self.optimizer = Optimizer(self.model.parameters(), **self.fine_tune_params['optimizer'])
        self.score_list, self.aug_score_list, self.gap_list, self.aug_gap_list = [], [], [], []

        # load dataset
        if tester_params['test_set_path'].endswith(".pkl"):
            self.test_data = torch.Tensor(load_dataset(tester_params['test_set_path'])[: self.tester_params['test_episodes']])
            opt_sol = load_dataset(tester_params['test_set_opt_sol_path'])[: self.tester_params['test_episodes']]  # [(obj, route), ...]
            self.opt_sol = [i[0] for i in opt_sol]
            env_params = {'problem_size': self.test_data.size(1), 'pomo_size': self.test_data.size(1)}
            self.env = Env(**env_params)  # we assume instances in the test/fine-tune dataset have the same problem size.
        else:
            # for solving instances with TSPLIB format
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            assert self.path_list[-1].endswith(".tsp")

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(">> Model loaded from {}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        start_time = time.time()

        if self.path_list:
            for path in self.path_list:
                self._solve_tsplib(path)
        else:
            if self.tester_params['test_robustness']:
                episode = 0
                test_data = self.test_data
                opt_sol = [0] * test_data.size(0)
                while episode < test_data.size(0):
                    remaining = test_data.size(0) - episode
                    batch_size = min(self.tester_params['test_batch_size'], remaining)
                    data = test_data[episode: episode + batch_size]
                    test_data[episode: episode + batch_size], opt_sol[episode: episode + batch_size] = self._generate_x_adv(data, eps=50.0, return_opt=True)
                    episode += batch_size
                self.test_data = test_data.cpu().numpy()
                self.opt_sol = [i[0] for i in opt_sol]
                # save the adv dataset
                filename = os.path.split(self.tester_params['test_set_path'])[-1]
                save_dataset(self.test_data, './adv_{}'.format(filename))
                save_dataset(opt_sol, './sol_adv_{}'.format(filename))
            if self.fine_tune_params['enable']:
                # fine-tune model on (little) data which has the same distribution of the test dataset (few-shot)
                start = self.tester_params['test_episodes']
                self.fine_tune_data = torch.Tensor(load_dataset(self.tester_params['test_set_path'])[start: start + self.fine_tune_params['fine_tune_episodes']])
                # self.fine_tune_data = augment_xy_data_by_8_fold(self.fine_tune_data)
                self._fine_tune_and_test()
            else:
                # test the model on test dataset (zero-shot)
                self._test(store_res=True)
            print(">> Evaluation on {} finished within {:.2f}s".format(self.tester_params['test_set_path'], time.time() - start_time))

            # save results to file
            with open(os.path.split(self.tester_params['test_set_path'])[-1], 'wb') as f:
                result = {"score_list": self.score_list, "aug_score_list": self.aug_score_list, "gap_list": self.gap_list, "aug_gap_list": self.aug_gap_list}
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                print(">> Save final results to {}".format(os.path.split(self.tester_params['test_set_path'])[-1]))

    def _test(self, store_res=True):
        self.time_estimator.reset()
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        assert len(self.test_data) == test_num_episode, "the number of test instances does not match!"
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            score, aug_score, all_score, all_aug_score = self._test_one_batch(self.test_data[episode: episode + batch_size])
            opt_sol = self.opt_sol[episode: episode + batch_size]
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            episode += batch_size
            gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            gap_AM.update(sum(gap)/batch_size, batch_size)
            aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)

            if store_res:
                self.score_list += all_score.tolist()
                self.aug_score_list += all_aug_score.tolist()
                self.gap_list += gap
                self.aug_gap_list += aug_gap

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(score_AM.avg, gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(aug_score_AM.avg, aug_gap_AM.avg))

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
        Evaluate few-shot generalization: fine-tune k steps on a small fine-tune dataset
        """
        fine_tune_episode = self.fine_tune_params['fine_tune_episodes']
        assert len(self.fine_tune_data) == fine_tune_episode, "the number of fine-tune instances does not match!"
        score_list, aug_score_list, gap_list, aug_gap_list = [], [], [], []

        for k in range(self.fine_tune_params['k']):
            # score, aug_score, gap, aug_gap = self._test(store_res=False)
            # score_list.append(score); aug_score_list.append(aug_score)
            # gap_list.append(gap); aug_gap_list.append(aug_gap)
            self.logger.info("Start fine-tune step {}".format(k+1))
            episode = 0
            while episode < fine_tune_episode:
                remaining = fine_tune_episode - episode
                batch_size = min(self.fine_tune_params['fine_tune_batch_size'], remaining)
                self._fine_tune_one_batch(self.fine_tune_data[episode:episode+batch_size])
                episode += batch_size

            # shuffle the data
            idx = torch.randperm(fine_tune_episode)
            self.fine_tune_data = self.fine_tune_data[idx]

        checkpoint_dict = {
            'epoch': k+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'result_log': None
        }
        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, k+1))

        score, aug_score, gap, aug_gap = self._test(store_res=True)
        score_list.append(score); aug_score_list.append(aug_score)
        gap_list.append(gap); aug_gap_list.append(aug_gap)

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
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
        # shape: (batch, augmentation * pomo)
        advantage = aug_reward - aug_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, augmentation * pomo)
        log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, self.env.pomo_size).permute(1, 0, 2).reshape(batch_size, -1)
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

    def _generate_x_adv(self, data, eps=10.0, return_opt=False):
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
        if return_opt:
            return data, solve_all_gurobi(data)

        return data

    def _solve_tsplib(self, path):
        """
        Solving one instance with TSPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n, 2]
        locations = torch.Tensor(original_locations / original_locations.max())  # Scale location coordinates to [0, 1]
        loc_scaler = original_locations.max()

        env_params = {'problem_size': locations.size(1), 'pomo_size': locations.size(1), 'loc_scaler': loc_scaler}
        self.env = Env(**env_params)
        _, _, no_aug_score, aug_score = self._test_one_batch(locations)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))
