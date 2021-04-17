import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize(histories, labels=None, type="val_accuracy", filename=None, start_from=0):
    fig = plt.figure()
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(histories) + 1)]
    for hist, label in zip(histories, labels):
        y = np.array(hist.history[type])
        plt.plot(range(start_from+1, len(y)+1), y[start_from:], label=label)
        plt.title(type)
        plt.xlabel("number of epoch")
        plt.ylabel(type)
        plt.legend()
    if filename:
        plt.savefig("plots/{0}_{1}".format(filename, type))
        plt.close(fig)
    else:
        plt.show()