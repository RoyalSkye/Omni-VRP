from utils.functions import seed_everything, load_problem
from utils.data_utils import save_dataset

def generate_train_task(opts):
    tasks_list = []
    if opts.variation_type == 'size':
        graph_sizes = [10, 20, 30, 50]
        if opts.problem == "tsp":
            pass
            # if opts.graph_size_continuous:
            #     graph_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == 'scale':
        scales = [[0, 1], [0, 2], [0, 4]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/SCALE_{}_{}-{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], int(task_prop['low']), int(task_prop['high']))
            tasks_list.append(task_prop)
    elif opts.variation_type == 'dist':
        for i in [1, 2, 5]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'low': 0, 'high': 1, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], task_prop['num_modes'])
            tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [1, 2, 5]:
            for cur_graph_size in [20, 30, 50]:
                num_modes = i
                task_prop = {'graph_size': cur_graph_size, 'low': 0, 'high': 1, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
                task_prop['insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], task_prop['num_modes'])
                tasks_list.append(task_prop)
    elif opts.variation_type == 'cap_vrp':
        train_tasks = [int(tsk) for tsk in (opts.train_tasks).split('_')]
        print("train_tasks ", train_tasks)
        for i in train_tasks:
            vrp_capacity = i
            task_prop = {'graph_size': opts.graph_size, 'vrp_capacity': vrp_capacity, 'low': 0, 'high': 1, 'variation_type': opts.variation_type}
            tasks_list.append(task_prop)
    elif opts.variation_type == "adv":
        for i in range(3):
            task_prop = {'graph_size': opts.graph_size, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': 'adv'}
            tasks_list.append(task_prop)
    elif opts.variation_type == "size_uniform":
        graph_sizes = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == "size_two_cluster":
        graph_sizes = [5, 6, 7, 8, 9, 91, 92, 93, 94, 95]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == "size_imbalanced":
        graph_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 95]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == "size_increasing_order":
        graph_sizes = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    elif opts.variation_type == "size_decreasing_order":
        graph_sizes = [95, 85, 75, 65, 55, 45, 35, 25, 15, 5]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'])
            tasks_list.append(task_prop)
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)

    return tasks_list


def generate_test_task(opts, test_seed=1234, fine_tune_seed=9999):
    tasks_list = []
    if opts.variation_type == 'size':
        graph_sizes = [10, 30, 50, 80, 100, 120, 150, 200]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "test", test_seed)
            task_prop['fine_tuning_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "fine_tuning", fine_tune_seed)
            tasks_list.append(task_prop)
    elif opts.variation_type == 'scale':
        scales = [[0.0, 3.0], [0.0, 5.0], [0.0, 8.0], [0.0, 10.0]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], task_prop['low'], task_prop['high'], "test", test_seed)
            task_prop['fine_tuning_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], task_prop['low'], task_prop['high'], "fine_tuning", fine_tune_seed)
            tasks_list.append(task_prop)
    elif opts.variation_type == 'dist':
        for i in [3, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'low': 0, 'high': 1, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", test_seed)
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", fine_tune_seed)
            tasks_list.append(task_prop)
    elif opts.variation_type == 'cap_vrp':
        for i in [20, 50]:
            vrp_capacity = i
            task_prop = {'graph_size': opts.graph_size, 'vrp_capacity': vrp_capacity, 'low': 0, 'high': 1, 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "test", test_seed)
            task_prop['fine_tuning_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "fine_tuning", fine_tune_seed)
            tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [3, 5, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'low': 0, 'high': 1, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", test_seed)
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", fine_tune_seed)
            tasks_list.append(task_prop)
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)

    return tasks_list


if __name__ == "__main__":
    from options import get_options
    opts = get_options()
    # opts.seed = 2023
    # opts.val_size = 10000

    opts.seed = 2022
    opts.fine_tune_size = 3000

    opts.problem = "tsp"
    opts.variation_type = "size"
    opts.graph_size = 40

    seed_everything(opts.seed)
    problem = load_problem(opts.problem)

    tasks_list = generate_test_task(opts, test_seed=2023, fine_tune_seed=2022)
    for task in tasks_list:
        # test_set = problem.make_dataset(num_samples=opts.val_size, filename=None, distribution=None, task=task)
        # save_dataset(test_set, "./new_data/size/tsp/" + task['test_dataset'])
        # print(task['test_dataset'], len(test_set))
        test_set = problem.make_dataset(num_samples=opts.fine_tune_size, filename=None, distribution=None, task=task)
        save_dataset(test_set, "./new_data/size/tsp/" + task['fine_tuning_dataset'])
        print(task['fine_tuning_dataset'], len(test_set))
