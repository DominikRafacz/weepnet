import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if filename:
        plt.savefig("plots/{0}".format(filename))
        plt.close(fig)
    return ax