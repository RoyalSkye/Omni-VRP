import math
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tqdm import tqdm

from source.TORCH_OBJECTS import *
from source.cvrp.model import multi_head_attention as multi_head_attention_CVRP
from source.cvrp.model import reshape_by_heads as reshape_by_heads_CVRP
from source.tsp.model import multi_head_attention as multi_head_attention_TSP
from source.tsp.model import reshape_by_heads as reshape_by_heads_TSP
AUG_S = 8


class prob_calc_added_layers_CVRP(nn.Module):
    """
    New nn.Module with added layers for the CVRP.
    """

    def __init__(self, batch_s, **model_params):
        super().__init__()

        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        # new added layer
        self.new = nn.Parameter(torch.zeros((batch_s, head_num * qkv_dim, head_num * qkv_dim), requires_grad=True))
        self.new_bias = nn.Parameter(torch.zeros((batch_s, 1, head_num * qkv_dim), requires_grad=True))
        self.new_2 = nn.Parameter(torch.zeros((batch_s, head_num * qkv_dim, head_num * qkv_dim), requires_grad=True))
        self.new_bias_2 = nn.Parameter(torch.zeros((batch_s, 1, head_num * qkv_dim), requires_grad=True))
        torch.nn.init.xavier_uniform_(self.new)
        torch.nn.init.xavier_uniform_(self.new_bias)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads_CVRP(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads_CVRP(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.single_head_key.requires_grad = True
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, ninf_mask=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        with torch.no_grad():
            input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
            # shape = (batch, group, EMBEDDING_DIM+1)
            q_last = reshape_by_heads_CVRP(self.Wq_last(input_cat), head_num=head_num)
            # shape: (batch, head_num, pomo, qkv_dim)
            # q = self.q1 + self.q2 + q_last
            # # shape: (batch, head_num, pomo, qkv_dim)
            q = q_last
            # shape: (batch, head_num, pomo, qkv_dim)
            out_concat = multi_head_attention_CVRP(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            # shape: (batch, pomo, head_num*qkv_dim)

        # Added layers start
        ###############################################
        residual = out_concat.detach()
        out_concat = F.relu(torch.matmul(out_concat, self.new) + self.new_bias.expand_as(out_concat))
        out_concat = torch.matmul(out_concat, self.new_2) + self.new_bias_2.expand_as(out_concat)
        out_concat += residual
        # shape = (batch, n, HEAD_NUM*KEY_DIM)
        # Added layers end
        ###############################################

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape: (batch, pomo, problem)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


class prob_calc_added_layers_TSP(nn.Module):
    """
    New nn.Module with added layers for the TSP.
    """

    def __init__(self, batch_s, **model_params):
        super().__init__()

        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        # new added layer
        self.new = nn.Parameter(torch.zeros((batch_s, head_num * qkv_dim, head_num * qkv_dim), requires_grad=True))
        self.new_bias = nn.Parameter(torch.zeros((batch_s, 1, head_num * qkv_dim), requires_grad=True))
        self.new_2 = nn.Parameter(torch.zeros((batch_s, head_num * qkv_dim, head_num * qkv_dim), requires_grad=True))
        self.new_bias_2 = nn.Parameter(torch.zeros((batch_s, 1, head_num * qkv_dim), requires_grad=True))
        torch.nn.init.xavier_uniform_(self.new)
        torch.nn.init.xavier_uniform_(self.new_bias)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads_TSP(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads_TSP(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads_TSP(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        with torch.no_grad():
            q_last = reshape_by_heads_TSP(self.Wq_last(encoded_last_node), head_num=head_num)
            # shape: (batch, head_num, pomo, qkv_dim)
            q = self.q_first + q_last
            # shape: (batch, head_num, pomo, qkv_dim)
            out_concat = multi_head_attention_TSP(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            # shape: (batch, pomo, head_num*qkv_dim)

        # Added layers start
        ###############################################
        residual = out_concat.detach()
        out_concat = F.relu(torch.matmul(out_concat, self.new) + self.new_bias.expand_as(out_concat))
        out_concat = torch.matmul(out_concat, self.new_2) + self.new_bias_2.expand_as(out_concat)
        out_concat += residual
        # Added layers end
        ###############################################

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape: (batch, pomo, problem)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


def replace_decoder(model, batch_s, state, problem, model_params):
    """Function to add layers to pretrained model while retaining weights from other layers."""

    # update decoder
    if problem == "CVRP":
        model.decoder = prob_calc_added_layers_CVRP(batch_s, **model_params)
    elif problem == "TSP":
        model.decoder = prob_calc_added_layers_TSP(batch_s, **model_params)
    model.decoder.load_state_dict(state_dict=state, strict=False)

    return model


#######################################################################################################################
# Search process
#######################################################################################################################

def run_eas_lay(model, instance_data, problem_size, config, get_episode_data_fn, augment_and_repeat_episode_data_fn):
    """
    Efficient active search using added layer updates
    """

    dataset_size = len(instance_data[0])

    assert config.batch_size <= dataset_size

    original_decoder_state_dict = model.decoder.state_dict()

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    for episode in range(math.ceil(dataset_size / config.batch_size)):

        print(">> {}: {}/{} instances finished.".format(config.method, episode * config.batch_size, dataset_size))
        episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        p_runs = config.p_runs  # Number of parallel runs per instance
        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size + 1  # Number of different rollouts per instance (+1 for incumbent solution construction)

        with torch.no_grad():
            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)
            # Replace the decoder of the loaded model with the modified decoder with added layers
            model_modified = replace_decoder(model, batch_s, original_decoder_state_dict, config.problem, config.model_params).cuda()
            group_state, reward, done = env.reset(group_size=group_s)
            # model_modified.reset(group_state)  # Generate the embeddings
            model_modified.pre_forward(group_state)

        # Only update the weights of the added layer during training
        optimizer = optim.Adam([model_modified.decoder.new, model_modified.decoder.new_2, model_modified.decoder.new_bias,
                                model_modified.decoder.new_bias_2], lr=config.param_lr, weight_decay=1e-6)

        incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

        # Start the search
        ###############################################
        t_start = time.time()
        for iter in range(config.max_iter):
            group_state, reward, done = env.reset(group_size=group_s)
            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1).repeat(p_runs, 1)

            # Start generating batch_s * group_s solutions
            ###############################################
            solutions = []

            step = 0
            if config.problem == "CVRP":
                # First Move is given
                first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                # model_modified(group_state, selected=first_action)  # do nothing for CVRP
                group_state, reward, done = env.step(first_action)
                solutions.append(first_action.unsqueeze(2))
                step += 1

            # First/Second Move is given
            second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
            if iter > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing the imitation learning loss
            model_modified(group_state, selected=second_action)  # for the first step, set_q1 for TSP, do nothing for CVRP
            group_state, reward, done = env.step(second_action)
            solutions.append(second_action.unsqueeze(2))
            step += 1

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = model_modified(group_state)
                # action_probs = model_modified.get_action_probabilities(group_state)
                # shape = (batch_s, group_s, problem)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if iter > 0:
                    action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing the imitation learning loss

                if config.problem == "CVRP":
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                batch_idx_mat = torch.arange(int(batch_s))[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if config.problem == "CVRP":
                    chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
                step += 1

            # Solution generation finished. Update incumbent solutions and best rewards
            ###############################################
            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            solutions = torch.cat(solutions, dim=2)
            if config.batch_size == 1:
                # Single instance search. Only a single incumbent solution exists that needs to be updated
                max_idx = torch.argmax(reward)
                best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                best_solution_iter = best_solution_iter[max_idx]
                incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                max_reward = reward.max()
            else:
                # Batch search. Update incumbent etc. separately for each instance
                max_reward, _ = group_reward.max(dim=2)
                max_reward, _ = max_reward.max(dim=0)

                reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
                iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)
                best_solutions_iter = torch.gather(solutions, 1, iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(1)
                incumbent_solutions[:, :best_solutions_iter.shape[1]] = best_solutions_iter

            # LEARNING - Actor
            # Use the same reinforcement learning method as during the training of the model
            # to update only the weights of the newly added layers
            ###############################################
            group_reward = reward[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch_s, group_s)
            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            loss_1 = group_loss.mean()  # Reinforcement learning loss
            loss_2 = -group_log_prob[:, group_s - 1].mean()  # Imitation learning loss
            loss = loss_1 + loss_2 * config.param_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(max_reward)

            if time.time() - t_start > config.max_runtime:
                break

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
        instance_costs[episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
