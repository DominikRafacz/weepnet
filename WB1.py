import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# params
img_size = (32, 32)
batch_size = 32
num_classes = 10
tf.random.set_seed(420)

# creating generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=20)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=420)
validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=420)

# original model https://www.kaggle.com/kedarsai/cifar-10-88-accuracy-using-keras
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps = int(train_generator.n / 64)
history = model.fit(train_generator, epochs=200, steps_per_epoch=steps, validation_data=validation_generator)

model.save("models/wb_cnn_kaggle")

# second model from https://www.kaggle.com/c/cifar-10/discussion/40237

baseMapNum = 32
weight_decay = 1e-4
batch_size = 64

train_datagen2 = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

train_generator2 = train_datagen2.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=420)
validation_generator2 = test_datagen.flow_from_directory(
    'data/validation',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=420)

model2 = Sequential()
model2.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.3))

model2.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.4))

model2.add(Flatten())
model2.add(Dense(num_classes, activation='softmax'))

model2.summary()

opt_rms = RMSprop(lr=0.0003, decay=1e-6)
model2.compile(loss='categorical_crossentropy',
               optimizer=opt_rms,
               metrics=['accuracy'])
history2 = model2.fit(train_generator2, epochs=125, verbose=1, validation_data=validation_generator2)
model2.save('models/wb_cnn_kaggle2')
