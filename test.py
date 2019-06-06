from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random

import tensorflow as tf


for filename in os.listdir('data/color/Train/'):
    X.append(img_to_array(load_img('data/color/Train/' + filename)))
    X = np.array(X, dtype=float)

# Set up train and test data

split = int(0.95 * len(X))
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain
model = Sequential()

model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Image transformerdatagen = ImageDataGenerator(        shear_range=0.2,        zoom_range=0.2,        rotation_range=20,        horizontal_flip=True)


# Generate training databatch_size = 10def image_a_b_gen(batch_size):    for batch in datagen.flow(Xtrain, batch_size=batch_size):        lab_batch = rgb2lab(batch)        X_batch = lab_batch[:,:,:,0]        Y_batch = lab_batch[:,:,:,1:] / 128        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model

tensorboard = TensorBoard(log_dir="data/color/output/first_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=1, steps_per_epoch=10)



# Save modelmodel_json = model.to_json()with open("model.json", "w") as json_file:    json_file.write(model_json) model.save_weights("model.h5")


# Test imagesXtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]

Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128

print(model.evaluate(Xtest, Ytest, batch_size=batch_size))
