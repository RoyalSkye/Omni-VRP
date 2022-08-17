
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
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_val_farthest_insertion.pkl".format(task_prop['graph_size'], task_prop['num_modes'])
            tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [1, 2, 5]:
            for cur_graph_size in [20, 30, 50]:
                num_modes = i
                task_prop = {'graph_size': cur_graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
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
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)

    return tasks_list


def generate_test_task(opts):
    tasks_list = []
    if opts.variation_type == 'size':
        graph_sizes = [80, 100, 120, 150]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'scale':
        scales = [[0.0, 3.0], [0.0, 5.0], [0.0, 8.0], [0.0, 10.0]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], task_prop['low'], task_prop['high'], "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], task_prop['low'], task_prop['high'], "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'dist':
        for i in [3, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'cap_vrp':
        for i in [20, 50]:
            vrp_capacity = i
            task_prop = {'graph_size': opts.graph_size, 'vrp_capacity': vrp_capacity, 'low': 0, 'high': 1, 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_cap_vrp_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(vrp_capacity), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    elif opts.variation_type == 'mix_dist_size':
        for i in [3, 5, 8]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm', 'variation_type': opts.variation_type}
            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "test", "1234")
            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem, task_prop['graph_size'], str(num_modes), "fine_tuning", "9999")
            tasks_list.append(task_prop)
    else:
        print("Invalid task distribution: opts.variation_type!")
        exit(0)

    return tasks_list
