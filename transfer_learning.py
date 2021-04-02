from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50V2, VGG19
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

img = Image.open('data/train/1.png')
img = np.array(img).reshape((32, 32, 3))

np.random.seed(420)
perm = np.random.permutation(50000)
inds_train = perm[0:5000]
inds_test = perm[5000:6000]

X_train = np.concatenate(
    [np.array(Image.open('data/train/' + str(i) + '.png')).reshape((1, 32, 32, 3)) for i in inds_train])
X_test = np.concatenate(
    [np.array(Image.open('data/test/' + str(i) + '.png')).reshape((1, 32, 32, 3)) for i in inds_test])

X_train = X_train / 255.0
X_test = X_test / 255.0
labels = pd.get_dummies(pd.read_csv('data/trainLabels.csv')['label'])

y_train = np.array(labels.iloc[inds_train, :])
y_test = np.array(labels.iloc[inds_test, :])

cifar10_dims = (32, 32, 3)

mdl = Sequential()

mdl.add(ResNet50V2(include_top=False, weights='imagenet', input_shape=cifar10_dims, classes=y_train.shape))
mdl.add(Flatten())
mdl.add(Dense(4000, activation='relu', input_dim=512))
mdl.add(Dense(2000, activation='relu'))
mdl.add(Dropout(.4))
mdl.add(Dense(1000, activation='relu'))
mdl.add(Dropout(.3))
mdl.add(Dense(500, activation='relu'))
mdl.add(Dropout(.2))
mdl.add(Dense(10, activation='softmax'))

lr = .001
batch_size= 100

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mdl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
mdl.fit(X_train, y_train, batch_size=batch_size, epochs=2, steps_per_epoch=X_train.shape[0]//batch_size, verbose=1, workers=4, use_multiprocessing=True)

mdl.summary()
