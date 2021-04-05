'''
https://www.kaggle.com/adi160/cifar-10-keras-transfer-learning
'''

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50V2, VGG19
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau

np.random.seed(420)
perm = np.random.permutation(50000)
inds_train = perm[0:40000]

X_train = np.concatenate(
    [np.array(Image.open('data/train/' + str(i + 1) + '.png')).reshape((1, 32, 32, 3)) for i in inds_train])
X_train = X_train / 255.0

labels = pd.get_dummies(pd.read_csv('data/trainLabels.csv')['label'])

y_train = np.array(labels.iloc[inds_train, :])
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.3)

# data augmentation
train_generator = ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    zoom_range=.1)

val_generator = ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    zoom_range=.1)

train_generator.fit(x_train)
val_generator.fit(x_val)

# learning rate annealer
lrr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.01,  # Factor by which learning rate will be reduced
    patience=3,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-5,
    verbose=1)  # The minimum learning rate

cifar10_dims = (32, 32, 3)

# resnet model

mdl = Sequential()

mdl.add(ResNet50V2(include_top=False, weights='imagenet', input_shape=cifar10_dims, classes=y_train.shape[1]))
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
batch_size = 100

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mdl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
mdl.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size),
                  epochs=80, steps_per_epoch=x_train.shape[0] // batch_size,
                  validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size),
                  callbacks=[lrr], verbose=1)

mdl.summary()
# plots of loss and accuracy
f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column

# Assign the first subplot to graph training loss and validation loss
ax[0].plot(mdl.history.history['loss'], color='b', label='Training Loss')
ax[0].plot(mdl.history.history['val_loss'], color='r', label='Validation Loss')

# Next lets plot the training accuracy and validation accuracy
ax[1].plot(mdl.history.history['accuracy'], color='b', label='Training  Accuracy')
ax[1].plot(mdl.history.history['val_accuracy'], color='r', label='Validation Accuracy')

# validation
y_pred = np.argmax(mdl.predict(x_val), axis=-1)
y_true = np.argmax(y_val, axis=1)
print(np.mean(y_pred == y_true))

cm = confusion_matrix(y_true, y_pred)

# saving model

mdl.save("models/transfer_learning1")
mdl.save_weights("models/transfer_learning1_weights")

# VGG model
lr = .001
batch_size = 100


mdl2 = Sequential()
mdl2.add(VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=y_train.shape[1]))
mdl2.add(Flatten())
mdl2.add(Dense(1024, activation=('relu'), input_dim=512))
mdl2.add(Dense(512, activation=('relu')))
mdl2.add(Dense(256, activation=('relu')))
mdl2.add(Dense(128, activation=('relu')))
mdl2.add(Dense(10, activation=('softmax')))

sgd = SGD(lr=lr, momentum=.9, nesterov=False)
mdl2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
mdl2.fit(train_generator.flow(x_train, y_train, batch_size=batch_size),
         epochs=40,
         steps_per_epoch=x_train.shape[0] // batch_size,
         validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size),
         callbacks=[lrr], verbose=1)

f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column

# Assign the first subplot to graph training loss and validation loss
ax[0].plot(mdl2.history.history['loss'], color='b', label='Training Loss')
ax[0].plot(mdl2.history.history['val_loss'], color='r', label='Validation Loss')

# Next lets plot the training accuracy and validation accuracy
ax[1].plot(mdl2.history.history['accuracy'], color='b', label='Training  Accuracy')
ax[1].plot(mdl2.history.history['val_accuracy'], color='r', label='Validation Accuracy')

mdl2.save("models/transfer_learning_vgg")
mdl2.save_weights("models/transfer_learning_vgg_weights")