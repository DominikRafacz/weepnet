import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# params
img_size = (32, 32)
batch_size = 64
num_classes = 10
tf.random.set_seed(420)
baseMapNum = 32
weight_decay = 1e-4

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

test_datagen = ImageDataGenerator(rescale=1. / 255)

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

model3 = Sequential([
    Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
           input_shape=(32, 32, 3)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(num_classes, activation='softmax')
])

model3.summary()

lrr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.5,  # Factor by which learning rate will be reduced
    patience=10,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-4,
    verbose=1)


opt_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model3.compile(loss='categorical_crossentropy',
               optimizer=opt_adam,
               metrics=['accuracy'])
history3 = model3.fit(train_generator2, epochs=125, verbose=1, validation_data=validation_generator2, callbacks=[lrr])
model3.evaluate(validation_generator2)
# Adam:
# loss - 0.57, accuracy - 0.872
# with lrr: loss - 0.464, accuracy - 0.889
pd.DataFrame(history3.history).plot()
f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column

# Assign the first subplot to graph training loss and validation loss
ax[0].plot(history3.history['loss'], color='b', label='Training Loss')
ax[0].plot(history3.history['val_loss'], color='r', label='Validation Loss')

# Next lets plot the training accuracy and validation accuracy
ax[1].plot(history3.history['accuracy'], color='b', label='Training  Accuracy')
ax[1].plot(history3.history['val_accuracy'], color='r', label='Validation Accuracy')
ax[1].plot(np.array(history3.history['lr']) * 1000, color='g', label='Validation Accuracy')
model3.save('models/wb_cnn_kaggle2_adam_with_lrr')

#################
model4 = Sequential([
    Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
           input_shape=(32, 32, 3)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(num_classes, activation='softmax')
])

model4.summary()

lrr2 = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.5,  # Factor by which learning rate will be reduced
    patience=5,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-4,
    verbose=1)

opt_rms = RMSprop(lr=0.005, decay=1e-6)
model4.compile(loss='categorical_crossentropy',
               optimizer=opt_adam,
               metrics=['accuracy'])
history4 = model4.fit(train_generator2, epochs=125, verbose=1, validation_data=validation_generator2, callbacks=[lrr2])
model4.evaluate(validation_generator2)
# loss - 0.493, accuracy - 0.8552
pd.DataFrame(history4.history).plot()
pd.DataFrame(history4.history["lr"]).plot()
history4_1 = model4.fit(train_generator2, epochs=50, verbose=1, validation_data=validation_generator2, callbacks=[lrr2])
# loss - 0.456, accuracy - 0.87
model4.evaluate(validation_generator2)
pd.DataFrame(history4_1.history).plot()