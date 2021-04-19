import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, History
from util import visualize

os.environ['TF_DETERMINISTIC_OPS'] = '1'
#
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# params
img_size = (32, 32)
batch_size = 64
num_classes = 10
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

reset_random_seeds(420)

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

# model3.summary()

lrr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.5,  # Factor by which learning rate will be reduced
    patience=10,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-4,
    verbose=1)

opt_adam = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
model3.compile(loss='categorical_crossentropy',
               optimizer=opt_adam,
               metrics=['accuracy'])
history3 = model3.fit(train_generator2, epochs=200, validation_data=validation_generator2, callbacks=[lrr])
model3.evaluate(validation_generator2)
# Adam:
# loss - 0.57, accuracy - 0.872
# with lrr: loss - 0.464, accuracy - 0.889
# on 200 epochs: loss - 0.462, accuracy - 0.887
pd.DataFrame(history3.history).plot()
visualize([history3])
visualize([history3], type="accuracy")
model3.save('models/wb_cnn_kaggle2_adam_with_lrr')
model3.save('models/wb_cnn_kaggle2_adam_with_lrr_200')
#################
reset_random_seeds(420)
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

# model4.summary()

lrr2 = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.5,  # Factor by which learning rate will be reduced
    patience=10,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-4,
    verbose=1)

opt_rms = RMSprop(lr=0.0005, decay=1e-6)
model4.compile(loss='categorical_crossentropy',
               optimizer=opt_adam,
               metrics=['accuracy'])
history4 = model4.fit(train_generator2, epochs=125, validation_data=validation_generator2, callbacks=[lrr2])
model4.evaluate(validation_generator2)
# loss - 0.457, accuracy - 0.865
pd.DataFrame(history4.history).plot()
pd.DataFrame(history4.history["lr"]).plot()
visualize([history3, history4], labels=["Adam", "RMSProp"], type="accuracy", filename="adam_vs_rmsp")
visualize([history3, history4], labels=["Adam", "RMSProp"], type="val_accuracy", filename="adam_vs_rmsp")
visualize([history3, history4], labels=["Adam", "RMSProp"], type="loss", filename="adam_vs_rmsp")
model4.save('models/wb_cnn_kaggle2_rmsp_with_lrr')
import pickle
with open('hist3.pickle', 'wb') as file_pi:
    pickle.dump(history3.history, file_pi)
with open('hist4.pickle', 'wb') as file_pi:
    pickle.dump(history4.history, file_pi)
####

sgd_opt = SGD(learning_rate=0.0005, momentum=0.9)