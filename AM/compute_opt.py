import os
import pickle
import argparse
from problems.tsp.problem_tsp import
from problems.tsp.tsp_gurobi import solve_all_gurobi
from utils import load_problem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute (near-)opt solution.")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--path', type=str, default="../new_data/size/tsp", help='Dataset file to use for validation')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset (default 0)')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to evaluate (default 10000)')

    opts = parser.parse_args(args)

    assert opts.problem == "tsp", "Unsupported currently!"

    files = os.listdir(opts.path)
    problem = load_problem(opts.problem)
    for file in files:
        if os.path.splitext(file)[-1][1:] not in ["pkl"]:
            print("Unsupported file detected: {}".format(file))
            continue
        print(">> Solving dataset {}".format(file))
        dataset = problem.make_dataset(filename=files, offset=0, num_samples=num_samples)
        opt_sol = solve_all_gurobi(dataset)
        # save results
        name = os.path.basename(file)[:-4] + "_opt.pkl"
        res_path = os.path.join(opts.path, name)
        with open(res_path, 'wb') as f:
            pickle.dump(opt_sol, f, pickle.HIGHEST_PROTOCOL)
