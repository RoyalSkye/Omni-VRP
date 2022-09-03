import warnings

import torch
import numpy as np
import os, random
import json
import pickle
import matplotlib.pyplot as plt


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


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
        return data


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
