import math
import time

import numpy as np
# from tqdm import tqdm

from source.TORCH_OBJECTS import *
AUG_S = 8


def run_eas_tab(model, instance_data, problem_size, config, get_episode_data_fn, augment_and_repeat_episode_data_fn):
    """
    Efficient active search using tabular updates
    """

    dataset_size = len(instance_data[0])

    assert config.batch_size <= dataset_size

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    for episode in range(math.ceil(dataset_size / config.batch_size)):

        print(">> {}: {}/{} instances finished.".format(config.method, episode * config.batch_size, dataset_size))
        episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        p_runs = config.p_runs  # Number of parallel runs per instance
        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size + 1  # Number of different rollouts per instance

        if config.problem == "TSP":
            nb_problem_nodes = problem_size
        elif config.problem == "CVRP":
            nb_problem_nodes = problem_size + 1

        with torch.no_grad():
            # Augment instances and generate the embeddings using the encoder
            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)
            group_state, reward, done = env.reset(group_size=group_s)
            # model.reset(group_state)  # Generate the embeddings
            model.pre_forward(group_state)

            # Initialize the matrix Q and other tensors for the incumbent solution tracking
            prob_matrix = torch.ones((batch_r, nb_problem_nodes * nb_problem_nodes), device="cuda")  # Matrix Q
            incumbent_edges = [[]] * batch_r
            incumbent_edges_probs = [[]] * batch_r
            max_reward = torch.full((batch_r,), -np.inf, device="cuda")
            incumbent_solutions = torch.zeros(batch_r, problem_size * 2, dtype=torch.long)

            # Start the search
            ###############################################
            t_start = time.time()
            for iter in range(config.max_iter):
                env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)  # not necessary?
                group_state, reward, done = env.reset(group_size=group_s)

                # First Move is given
                if config.problem == "TSP":  # start from nodes
                    first_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
                    model(group_state, selected=first_action)  # for the first step, set_q1
                    group_state, reward, done = env.step(first_action)
                    last_action = first_action
                elif config.problem == "CVRP":  # start from node_0-depot
                    first_action = LongTensor(np.zeros((batch_s, group_s)))
                    # model(group_state, selected=first_action)  # do nothing for CVRP
                    group_state, reward, done = env.step(first_action)
                    last_action = first_action

                # Expand Q because the same Q matrix is used for all 8 augmentations of an instance
                prob_matrix_expanded = prob_matrix.repeat(AUG_S, 1).unsqueeze(1).expand(batch_s, group_s, -1).reshape(batch_s, group_s, nb_problem_nodes, nb_problem_nodes)

                # Construct solutions
                ###############################################
                solutions = [first_action.unsqueeze(2)]
                solutions_iter_edges_prob = []

                while not done:
                    action_probs_nn = model(group_state)
                    # action_probs_nn = model.get_action_probabilities(group_state)  # Get probabilities for the next action from the model

                    # Get corresponding values for the next action from the Q matrix
                    idx = last_action.unsqueeze(2).unsqueeze(3).expand(batch_s, group_s, 1, nb_problem_nodes)
                    matrix_probs = torch.gather(prob_matrix_expanded, 2, idx).squeeze()

                    # Calculate the final probabilities by combining both values
                    action_probs = action_probs_nn ** config.param_alpha * matrix_probs

                    # Sample action
                    action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
                    if config.problem == "CVRP":
                        action[group_state.finished] = 0  # stay at depot, if you are finished
                    group_state, reward, done = env.step(action)
                    solutions.append(action.unsqueeze(2))
                    solutions_iter_edges_prob.append(torch.gather(action_probs_nn, 2, action.unsqueeze(2)))

                    last_action = action

                # Solution generation finished. Update incumbent solutions, best rewards and the Q matrix
                ###############################################
                group_reward = reward.reshape(AUG_S, int(batch_s / AUG_S), group_s)
                solutions = torch.cat(solutions, dim=2)
                solutions_iter_edges_prob = torch.cat(solutions_iter_edges_prob, dim=2)
                max_reward_iter, _ = group_reward.max(dim=2)
                max_reward_iter, _ = max_reward_iter.max(dim=0)
                improved_idx = max_reward < max_reward_iter

                if improved_idx.any():
                    # Update incumbent rewards of the search
                    max_reward[improved_idx] = max_reward_iter[improved_idx]
                    # Find the best solutions per instance (over all augmentations)
                    reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)[improved_idx]
                    iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                    solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                    solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)[[improved_idx]]
                    best_solutions_iter = torch.gather(solutions, 1, iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(1)

                    # Update incumbent solution storage
                    incumbent_solutions[improved_idx, :best_solutions_iter.shape[1]] = best_solutions_iter.cpu()

                    # Find the edge ids that are part of best solutions of this iteration
                    best_solutions_iter = best_solutions_iter.unsqueeze(1).unsqueeze(3)
                    best_solutions_iter_edges = torch.cat([best_solutions_iter[:, :, :-1], best_solutions_iter[:, :, 1:]], dim=3)
                    best_solutions_iter_edges = best_solutions_iter_edges[:, :, :, 0] * nb_problem_nodes + best_solutions_iter_edges[:, :, :, 1]

                    # Find the probability assigned to each edge of the incumbent solutions by the network
                    solutions_iter_edges_prob = solutions_iter_edges_prob.reshape(AUG_S, round(batch_s / AUG_S), group_s, -1)
                    solutions_iter_edges_prob = solutions_iter_edges_prob.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)[[improved_idx]]
                    best_solutions_iter_edge_prob = torch.gather(solutions_iter_edges_prob, 1, iter_best_k.unsqueeze(2).expand(-1, -1, solutions_iter_edges_prob.shape[2]))

                    # Update the incumbent edges and their probability
                    for j, idx in enumerate(improved_idx.nonzero()):
                        incumbent_edges[idx] = best_solutions_iter_edges[j, 0]
                        incumbent_edges_probs[idx] = best_solutions_iter_edge_prob.squeeze(1)[j]

                    # Update the matrix Q based on the incumbent edges and their probability
                    prob_matrix *= 0
                    for i in range(batch_r):
                        prob_matrix[i, incumbent_edges[i]] = (config.param_sigma / (incumbent_edges_probs[i] ** config.param_alpha))
                    prob_matrix = torch.clamp(prob_matrix, 1)

                if config.p_runs > 1:
                    # Single instance search with multiple runs per instance. In this case the
                    # matrix Q of the incumbent is used to influence the other matrices (see Appendix A of the paper)
                    beta = 1 - min(1 - iter / config.max_iter, 1 - (time.time() - t_start) / config.max_runtime)
                    best_prob_matrix = prob_matrix[torch.argmax(max_reward)]
                    prob_matrix = prob_matrix * (1 - beta) + best_prob_matrix * beta

                if time.time() - t_start > config.max_runtime:
                    break

        # If a single instance is solved with multiple runs, only the best solutions is returned
        if p_runs > 1:
            incumbent_solutions = incumbent_solutions[torch.argmax(max_reward)]
            max_reward = max_reward.max()

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
        instance_costs[episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
