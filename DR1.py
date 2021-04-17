'''
testing other transfer learning architectures

https://www.kaggle.com/adi160/cifar-10-keras-transfer-learning
'''

import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from util import visualize
import csv

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 666
random.seed(666)
np.random.seed(666)
tf.random.set_seed(666)

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
batch_size = 128

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

adam = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

results = [None] * 3

for i in range(3):
    mdl.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    results[i] = mdl.fit(train_generator,
                         epochs=100,
                         validation_data=validation_generator,
                         callbacks=[lrr],
                         verbose=1)

visualize(results, filename='resnet_val')

for i, history in enumerate(results):
    with open('histories/resnet_' + str(i) + '_history.csv', 'x') as f:
        wrtr = csv.DictWriter(f, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
        wrtr.writeheader()
        wrtr.writerows([
            {'loss': history.history['loss'][i],
             'val_loss': history.history['val_loss'][i],
             'accuracy': history.history['accuracy'][i],
             'val_accuracy': history.history['val_accuracy'][i]
             } for i in range(len(history.history['loss']))
        ])
