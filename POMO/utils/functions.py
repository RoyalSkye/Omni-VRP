import warnings

import torch
import numpy as np
import os, random
import math
import json
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from scipy.stats import ttest_rel


def check_null_hypothesis(a, b):
    print(len(a), a)
    print(len(b), b)
    alpha_threshold = 0.05
    t, p = ttest_rel(a, b)  # Calc p value
    print(t, p)
    p_val = p / 2  # one-sided
    # assert t < 0, "T-statistic should be negative"
    print("p-value: {}".format(p_val))
    if p_val < alpha_threshold:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Rejected.")
    else:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Accepted.")


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename, disable_print=False):

    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
        return data


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param/1e6))


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def show(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=300):
    plt.style.use('fast')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(8, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            plt.scatter(x[i], y[i], color=colors[i], s=50)  # label=label[i]
            # plt.plot(x[i], y[i], color=colors[i], label=label[i], marker='o', ms=10)
        else:
            plt.scatter(x[i], y[i], color=colors[i % len(label)])
            # plt.plot(x[i], y[i], color=colors[i % len(label)], marker='o', ms=10)

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    plt.title(title, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper right', fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")
