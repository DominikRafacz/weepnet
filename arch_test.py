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
from util import visualize, visualize2

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


def create_model(seed, epochs, dense):
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

        Flatten()
    ])
    for i in range(dense):
        model.add(Dense(128//(i+1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))
    lrr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=.5,
        patience=8,
        min_lr=1e-4,
        verbose=1)
    opt_adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_adam,
                  metrics=['accuracy'])
    history = model.fit(train_generator2, epochs=epochs, validation_data=validation_generator2, callbacks=[lrr])
    loss, acc = model.evaluate(validation_generator2)
    return model, history, loss, acc


models = []
histories = []
losses = []
accs = []
for d in [1, 2]:
    for seed in [420, 42, 402]:
        print("Added dense {0}, seed {1}".format(d, seed))
        model, hist, loss, acc = create_model(seed, 50, d)
        models.append(model)
        histories.append(hist)
        losses.append(loss)
        accs.append(acc)

print("1 dense layer avg loss: {0:.4f}, avg accuracy: {1:.4f}".format(np.mean(losses[:3]), np.mean(accs[:3])))
print("2 dense layers avg loss: {0:.4f}, avg accuracy: {1:.4f}".format(np.mean(losses[3:6]), np.mean(accs[3:6])))

df = pd.DataFrame({"loss": losses, "acc": accs})
df.to_csv("arch_comparison.csv")

with open("lr_hist.pickle", "rb") as f:
    hist_lr = pickle.load(f)[3:]

histories_arch = [histories[i].history for i in range(len(histories))]
with open("arch_hist.pickle", "wb") as f:
    pickle.dump(histories_arch, f)
histories3 = hist_lr + histories_arch

labels = list(np.array([[name + " " +str(i) for i in range(1, 4)] for name in ["0 dense layers", "1 dense layer", "2 dense layers" ]]).flatten())
visualize2(histories3, labels=labels, type="accuracy", filename="arch_comparison",
          title="Comparison of accuracy on training set")
visualize2(histories3, labels=labels, type="loss", filename="arch_comparison",
          title="Comparison of loss on training set")
visualize2(histories3, labels=labels, type="val_accuracy", filename="arch_comparison",
          title="Comparison of accuracy on validation set")
visualize2(histories3, labels=labels, type="val_loss", filename="arch_comparison",
          title="Comparison of loss on validation set")

visualize2(histories3, labels=labels, type="accuracy", filename="arch_comparison2",
          title="Comparison of accuracy on training set", start_from=20)
visualize2(histories3, labels=labels, type="loss", filename="arch_comparison2",
          title="Comparison of loss on training set", start_from=20)
visualize2(histories3, labels=labels, type="val_accuracy", filename="arch_comparison2",
          title="Comparison of accuracy on validation set", start_from=20)
visualize2(histories3, labels=labels, type="val_loss", filename="arch_comparison2",
          title="Comparison of loss on validation set", start_from=20)

visualize2(histories3,labels=labels, type="lr", title="Changes of learning rate by callback",filename="arch_comparison")
