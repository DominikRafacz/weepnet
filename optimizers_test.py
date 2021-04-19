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
from util import visualize

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
batch_size = 64
num_classes = 10
baseMapNum = 32
weight_decay = 1e-4


def create_model(seed, optim, epochs):
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
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    history = model.fit(train_generator2, epochs=epochs, validation_data=validation_generator2)
    loss, acc = model.evaluate(validation_generator2)
    return model, history, loss, acc


opt_rms = RMSprop(lr=0.0003, decay=1e-6)
opt_adam = Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999)
opt_sgd = SGD(learning_rate=0.0003, momentum=0.9)

models = []
histories = []
losses = []
accs = []
for opt in [opt_rms, opt_adam, opt_sgd]:
    for seed in [420, 42, 402]:
        print("{0}, seed {1}".format(opt._name, seed))
        model, hist, loss, acc = create_model(seed, opt, 50)
        models.append(model)
        histories.append(hist)
        losses.append(loss)
        accs.append(acc)

print("RMSProp avg loss: {0:.4f}, avg accuracy: {1:.4f}".format(np.mean(losses[:3]), np.mean(accs[:3])))
print("Adam avg loss: {0:.4f}, avg accuracy: {1:.4f}".format(np.mean(losses[3:6]), np.mean(accs[3:6])))
print("SGD avg loss: {0:.4f}, avg accuracy: {1:.4f}".format(np.mean(losses[6:]), np.mean(accs[6:])))
df = pd.DataFrame({"loss": losses, "acc": accs})
df.to_csv("optim_comparison.csv")
labels = list(np.array([[opt._name + str(i) for i in range(1,4)] for opt in [opt_rms, opt_adam, opt_sgd]]).flatten())
visualize(histories, labels=labels, type="accuracy", filename="optim_comparison", title="Comparison of accuracy on training set")
visualize(histories, labels=labels, type="loss", filename="optim_comparison", title="Comparison of loss on training set")
visualize(histories, labels=labels, type="val_accuracy", filename="optim_comparison", title="Comparison of accuracy on validation set")
visualize(histories, labels=labels, type="val_loss", filename="optim_comparison", title="Comparison of loss on validation set")

histories2 = [histories[i].history for i in range(len(histories))]
with open("optim_hist.pickle", "wb") as f:
    pickle.dump(histories2, f)
