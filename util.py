import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize(histories, labels=None, type="val_accuracy", filename=None, start_from=0, title=None):
    fig = plt.figure()
    if not title:
        title = type
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(histories) + 1)]
    for hist, label in zip(histories, labels):
        y = np.array(hist.history[type])
        plt.plot(range(start_from + 1, len(y) + 1), y[start_from:], label=label)
        plt.title(title)
        plt.xlabel("number of epoch")
        plt.ylabel(type)
        plt.legend()
    if filename:
        plt.savefig("plots/{0}_{1}".format(filename, type))
        plt.close(fig)
    else:
        plt.show()


def visualize2(histories, labels=None, type="val_accuracy", filename=None, start_from=0, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(histories)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    if not title:
        title = type
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(histories) + 1)]
    for hist, label in zip(histories, labels):
        y = np.array(hist[type])
        plt.plot(range(start_from + 1, len(y) + 1), y[start_from:], label=label)
        plt.title(title)
        plt.xlabel("number of epoch")
        plt.ylabel(type)
        plt.legend()
    if filename:
        plt.savefig("plots/{0}_{1}".format(filename, type))
        plt.close(fig)
    else:
        plt.show()