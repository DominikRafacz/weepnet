import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, History
import time

# determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# generators
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

# params
img_size = (32, 32)
num_classes = 10
baseMapNum = 32
weight_decay = 1e-4


def create_model(seed, epochs, batch_size):
    train_generator2 = train_datagen2.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed)
    validation_generator2 = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed)

    reset_random_seeds(seed)
    model = Sequential([
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
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    lrr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=.5,
        patience=10,
        min_lr=1e-4,
        verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    opt_adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_adam,
                  metrics=['accuracy'])
    history = model.fit(train_generator2, epochs=epochs, validation_data=validation_generator2, callbacks=[lrr, es])
    loss, acc = model.evaluate(validation_generator2)
    return model, history, loss, acc


t1 = time.time()
model, hist, loss, acc = create_model(84, 200, 64)
t2 = time.time()
print("Time: {0:.3f}".format(t2-t1))

model.save("models/final_model")

test_generator = test_datagen.flow_from_directory(
        'data/',
        target_size=img_size,
        classes=['test'],
        batch_size=1,
        class_mode=None,
        shuffle=False)

pred = model.predict(test_generator, verbose=1)

classes = np.argmax(pred, axis=1)
classes_dict = dict(list(zip(range(10), os.listdir("data/train"))))
classes_names = [classes_dict[c] for c in classes]
pred_df = pd.DataFrame(classes_names, columns=["label"])

filenames = list(test_generator.filenames)
ids = [f.split(".")[0][5:] for f in filenames]
pred_df.index = ids
pred_df.to_csv("kaggle_pred1.csv", index_label="id")

#######
validation_generator_pred = test_datagen.flow_from_directory(
        'data/validation',
        target_size=img_size,
        batch_size=1,
        class_mode="categorical",
        shuffle=False)

pred_valid = model.predict(validation_generator_pred, verbose=1)

pred_valid_classes = np.argmax(pred_valid, axis=1)

plot_confusion_matrix(validation_generator_pred.classes, pred_valid_classes, os.listdir("data/train"), normalize=True, filename="conf_matrix_valid")