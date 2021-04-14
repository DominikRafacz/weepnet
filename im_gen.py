import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

img_size = (32, 32)
batch_size = 128
# build a structure for imagedatagenerator
# img_dir = "data/train"
# labels = pd.read_csv("data/trainLabels.csv").loc[:, 'label']
# np.random.seed(420)
# perm = np.random.permutation(50000)
# inds_train = perm[0:40000]
# inds_val = perm[40000:50000]
# for ind in inds_train:
#     cls = labels[ind]
#     if not os.path.exists("data/train/{}".format(cls)):
#         os.makedirs("data/train/{}".format(cls))
#     if os.path.exists("data/train/{}.png".format(ind+1)):
#         os.rename("data/train/{}.png".format(ind+1), "data/train/{0}/{1}.png".format(cls, ind+1))
# for ind in inds_val:
#     cls = labels[ind]
#     if not os.path.exists("data/validation/{}".format(cls)):
#         os.makedirs("data/validation/{}".format(cls))
#     if os.path.exists("data/train/{}.png".format(ind+1)):
#         os.rename("data/train/{}.png".format(ind+1), "data/validation/{0}/{1}.png".format(cls, ind+1))

# creating generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        seed=420)
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        seed=420)

num_classes = 10

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

# learning rate annealer
lrr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to be measured
    factor=.5,  # Factor by which learning rate will be reduced
    patience=5,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
    min_lr=1e-4,
    verbose=1)

adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, epochs=200, callbacks=[lrr])

model.evaluate(validation_generator)
pd.DataFrame(history.history['lr']).plot()

model.save("models/custom_cnn1")
model.save_weights("models/custom_cnn1_weights")

###################################################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=30,
    shear_range=0.1)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        seed=420)
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        seed=420)

model2 = Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu',  padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

es = EarlyStopping(monitor="val_loss", patience=5)
model2.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[es])

model2.save("models/custom_cnn2")
model2.save_weights("models/custom_cnn2_weights")