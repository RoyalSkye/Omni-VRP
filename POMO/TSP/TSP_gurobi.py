import os, sys
import time
import glob
import pickle
import argparse
import numpy as np
from gurobipy import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils.functions import seed_everything, load_dataset, save_dataset


def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Copyright 2017, Gurobi Optimization, Inc.
    Solve a traveling salesman problem on a set of
    points using lazy constraints.   The base MIP model only includes
    'degree-2' constraints, requiring each node to have exactly
    two incident edges.  Solutions to this model may contain subtours -
    tours that don't visit every city.  The lazy constraint callback
    adds new constraints to cut them off.

    Solves the Euclidan TSP problem to optimality using the MIP formulation
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate
    :return:
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour


def solve_all_gurobi(dataset):
    results = []
    for i, instance in enumerate(dataset):
        print("Solving instance {}".format(i))
        # some hard instances may take prohibitively long time, and ultimately kill the solver, so we set tl=1800s for TSP100 to avoid that.
        result = solve_euclidian_tsp(instance)
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute (near-)opt solution.")
    parser.add_argument('--baseline', type=str, default='gurobi', choices=['gurobi', 'lkh3', 'concorde', 'farthest_insertion'], help="which baseline to use")
    parser.add_argument('--path', type=str, default="../../data/TSP", help='Dataset file')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset (default 0)')
    parser.add_argument('--timelimit', type=int, default=0, help='(total) time limit for baseline')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to evaluate (default 10000)')

    args = parser.parse_args()

    # Note: we only solve [0:10000] instances for testing
    file_names = glob.glob(os.path.join(args.path, "*.pkl"))
    for file_name in file_names:
        data = load_dataset(file_name)
        print(">> {}: Solving dataset {}".format(args.baseline, file_name))
        start_time = time.time()
        if args.baseline == "gurobi":
            res = solve_all_gurobi(data[args.offset:args.offset+args.num_samples])  # [(obj, route), ...]
            print(">> Completed within {}s".format(time.time() - start_time))
            # save the results
            path = os.path.join(args.path, args.baseline)
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, os.path.split(file_name)[-1])
            with open(path, "wb") as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError
