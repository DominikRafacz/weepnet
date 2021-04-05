'''
https://www.kaggle.com/kedarsai/cifar-10-88-accuracy-using-keras
'''
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, AveragePooling2D, Activation, \
    Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(420)
perm = np.random.permutation(50000)
inds_train = perm[0:40000]

X_train = np.concatenate(
    [np.array(Image.open('data/train/' + str(i + 1) + '.png')).reshape((1, 32, 32, 3)) for i in inds_train])
X_train = X_train / 255.0

labels = pd.get_dummies(pd.read_csv('data/trainLabels.csv')['label'])

y_train = np.array(labels.iloc[inds_train, :])
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.3)

model6 = Sequential()
model6.add(
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model6.add(BatchNormalization())
model6.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2, 2)))
model6.add(Dropout(0.2))
model6.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model6.add(BatchNormalization())
model6.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2, 2)))
model6.add(Dropout(0.3))
model6.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model6.add(BatchNormalization())
model6.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2, 2)))
model6.add(Dropout(0.4))
model6.add(Flatten())
model6.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model6.add(BatchNormalization())
model6.add(Dropout(0.5))
model6.add(Dense(10, activation='softmax'))
# compile model
# opt = SGD(lr=0.001, momentum=0.9)
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image Data Generator , we are shifting image accross width and height also we are flipping the image horizantally.
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=20)
it_train = datagen.flow(x_train, y_train)
steps = int(x_train.shape[0] / 64)
history6 = model6.fit_generator(it_train, epochs=200, steps_per_epoch=steps, validation_data=(x_val, y_val))

evaluation = model6.evaluate(x_val, y_val)
print('Test Accuracy: {}'.format(evaluation[1]))