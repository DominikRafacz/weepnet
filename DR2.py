from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import vgg19
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from util import visualize
import csv

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 666
random.seed(666)
np.random.seed(666)
tf.random.set_seed(666)

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2,
    preprocessing_function=vgg19.preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=vgg19.preprocess_input)

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

lrr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.01,  # Factor by which learning rate will be reduced
    patience=3,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-5,
    verbose=1)  # The minimum learning rate

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

results = [None] * 3

for i in range(3):
    mdl2.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    results[i] = mdl2.fit(train_generator,
                          epochs=100,
                          validation_data=validation_generator,
                          callbacks=[lrr], verbose=1)

visualize(results, filename='vgg_val')

for i, history in enumerate(results):
    with open('histories/vgg_' + str(i) + '_history.csv', 'x') as f:
        wrtr = csv.DictWriter(f, ['loss', 'val_loss', 'accuracy', 'val_accuracy'])
        wrtr.writeheader()
        wrtr.writerows([
            {'loss': history.history['loss'][i],
             'val_loss': history.history['val_loss'][i],
             'accuracy': history.history['accuracy'][i],
             'val_accuracy': history.history['val_accuracy'][i]
             } for i in range(len(history.history['loss']))
        ])
