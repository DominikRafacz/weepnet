from PIL import Image
import numpy as np
import pandas as pd

from deepnet.nnet import CNN
from deepnet.solver import sgd_momentum
from run_cnn import make_cifar10_cnn

img = Image.open('data/cifar10/train/1.png')
np.array(img).reshape((3, 32, 32))

np.random.seed(420)
perm = np.random.permutation(50000)
inds_train = perm[0:5000]
inds_test = perm[5000:6000]

X_train = np.concatenate([np.array(Image.open('data/cifar10/train/' + str(i) + '.png')).reshape((1, 3, 32, 32)) for i in inds_train])
X_test = np.concatenate([np.array(Image.open('data/cifar10/test/' + str(i) + '.png')).reshape((1, 3, 32, 32)) for i in inds_test])

X_train = X_train / 255.0
X_test = X_test / 255.0


y_train = np.array(pd.get_dummies(pd.read_csv('data/cifar10/trainLabels.csv')['label']))[inds_train, :].argmax(axis=1)
y_test = np.array(pd.get_dummies(pd.read_csv('data/cifar10/trainLabels.csv')['label']))[inds_test, :].argmax(axis=1)

cifar10_dims = (3, 32, 32)

cnn = CNN(make_cifar10_cnn(cifar10_dims, num_class=10))
cnn = sgd_momentum(cnn, X_train, y_train, minibatch_size=100, epoch=200, learning_rate=0.001, X_test=X_test, y_test=y_test)
