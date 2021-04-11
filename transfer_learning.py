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
from tensorflow.keras.applications import vgg19
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau



# data augmentation
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

img_size = (32, 32)
batch_size = 100

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse')

# x, y = train_generator.next()
# i = 7
# plt.figure()
# plt.imshow(x[i])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# print(y[i])
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
mdl.add(ResNet50V2(include_top=False, weights='imagenet', input_shape=cifar10_dims, classes=10))
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

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mdl.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = mdl.fit(train_generator,
                  epochs=100,
                  validation_data=validation_generator,
                  callbacks=[lrr],
                  verbose=1)

# mdl.summary()
# # plots of loss and accuracy
# f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column
#
# # Assign the first subplot to graph training loss and validation loss
# ax[0].plot(hist.history['loss'], color='b', label='Training Loss')
# ax[0].plot(hist.history['val_loss'], color='r', label='Validation Loss')
#
# # Next lets plot the training accuracy and validation accuracy
# ax[1].plot(hist.history['accuracy'], color='b', label='Training  Accuracy')
# ax[1].plot(hist.history['val_accuracy'], color='r', label='Validation Accuracy')

# validation
# mdl.evaluate(validation_generator)
#
#
# test_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=img_size,
#         batch_size=1,
#         class_mode='sparse',
#         shuffle=False)
#
# y_pred = np.argmax(mdl.predict(test_generator), axis=-1)
#
# y_true = pd.get_dummies(pd.Series(test_generator.classes)).idxmax(axis=1)
# print(np.mean(y_pred == y_true))
#
# cm = confusion_matrix(y_true, y_pred)

# saving model

mdl.save("models/transfer_learning_renet50")
mdl.save_weights("models/transfer_learning_resnet50_weights")

# VGG model

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2,
    preprocessing_function=vgg19.preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=vgg19.preprocess_input)

img_size = (32, 32)
batch_size = 100

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse')

lr = .001

mdl2 = Sequential()
mdl2.add(VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=10))
mdl2.add(Flatten())
mdl2.add(Dense(1024, activation='relu', input_dim=512))
mdl2.add(Dense(512, activation='relu'))
mdl2.add(Dense(256, activation='relu'))
mdl2.add(Dense(128, activation='relu'))
mdl2.add(Dense(10, activation='softmax'))

sgd = SGD(lr=lr, momentum=.9, nesterov=False)
mdl2.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist2 = mdl2.fit(train_generator,
         epochs=40,
         validation_data=validation_generator,
         callbacks=[lrr], verbose=1)


# f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column
#
# # Assign the first subplot to graph training loss and validation loss
# ax[0].plot(hist2.history['loss'], color='b', label='Training Loss')
# ax[0].plot(hist2.history['val_loss'], color='r', label='Validation Loss')
#
# # Next lets plot the training accuracy and validation accuracy
# ax[1].plot(hist2.history['accuracy'], color='b', label='Training  Accuracy')
# ax[1].plot(hist2.history['val_accuracy'], color='r', label='Validation Accuracy')

mdl2.save("models/transfer_learning_vgg19")
mdl2.save_weights("models/transfer_learning_vgg19_weights")